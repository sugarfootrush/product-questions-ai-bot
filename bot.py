import os
import logging
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from typing import Tuple, Optional, List, Dict, Any
import google.generativeai as genai
import chromadb

# --- Global Configuration ---

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the Gemini API key
try:
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("CRITICAL: GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=gemini_api_key)
    logger.info("Gemini API configured successfully.")
except ValueError as e:
    logger.critical(e)
    exit(1) # Exit if critical config is missing
except Exception as e:
    logger.critical(f"CRITICAL: Error configuring Gemini: {e}")
    exit(1)

# ChromaDB setup for querying
CHROMA_DATA_PATH = "chroma_data"
COLLECTION_NAME = "slack_history"
EMBEDDING_MODEL_NAME_FOR_QUERY = "models/embedding-001" # Must match model used for populating

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    slack_history_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    # You could also use get_or_create_collection if you want the bot to be able to create it,
    # but for querying an existing populated DB, get_collection is fine.
    # If get_collection fails, it means the DB or collection wasn't created correctly.
    logger.info(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}' with {slack_history_collection.count()} items.")
except Exception as e:
    logger.error(f"Error connecting to ChromaDB collection '{COLLECTION_NAME}': {e}. Please ensure vector_db_populator.py ran successfully.", exc_info=True)
    # Depending on severity, you might want the bot to exit or operate in a degraded mode.
    # For now, it will log the error and the bot will try to run, but search will fail.
    slack_history_collection = None # Ensure it's defined so later checks don't cause NameError

def search_slack_history(user_question: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Searches the Slack history vector DB for messages relevant to the user's question.
    """
    if not slack_history_collection:
        logger.error("Slack history collection (ChromaDB) is not available for searching.")
        return []
    if not user_question:
        return []

    try:
        logger.info(f"Generating embedding for user query: '{user_question}'")
        # For querying, the task_type is "RETRIEVAL_QUERY"
        query_embedding_result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME_FOR_QUERY,
            content=user_question,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = query_embedding_result['embedding']
        
        logger.info(f"Querying ChromaDB collection '{COLLECTION_NAME}' for top {top_n} results.")
        results = slack_history_collection.query(
            query_embeddings=[query_embedding], # Query expects a list of embeddings
            n_results=top_n,
            include=['documents', 'metadatas', 'distances'] # Ask for docs, metadata, and distances
        )
        logger.debug(f"ChromaDB query results: {results}")

        # Process results. ChromaDB returns lists of lists for each included item,
        # corresponding to each query embedding. Since we only have one query embedding,
        # we access the first element (index 0) of these lists.
        found_messages = []
        if results and results.get('ids') and results.get('ids')[0]: # Check if 'ids' and its first list exist
            for i, doc_id in enumerate(results['ids'][0]):
                message_data = {
                    "id": doc_id,
                    "text": results['documents'][0][i] if results.get('documents') else "N/A",
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "distance": results['distances'][0][i] if results.get('distances') else float('inf')
                }
                # Add permalink from metadata if available
                message_data["permalink"] = message_data.get("metadata", {}).get("permalink", "")
                found_messages.append(message_data)
            
            # Sort by distance (lower is better) if not already sorted (Chroma usually does)
            found_messages.sort(key=lambda x: x['distance'])
            logger.info(f"Found {len(found_messages)} relevant messages in Slack history.")
            return found_messages
        else:
            logger.info("No relevant messages found in Slack history for the query.")
            return []

    except Exception as e:
        logger.error(f"Error during Slack history search for query '{user_question}': {e}", exc_info=True)
        return []
    

# Initialize the Gemini model
MODEL_NAME = 'gemini-1.5-flash-latest' # Or 'gemini-pro'
try:
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Gemini model '{MODEL_NAME}' initialized successfully.")
except Exception as e:
    logger.critical(f"CRITICAL: Error initializing Gemini model '{MODEL_NAME}': {e}")
    exit(1)

# --- Utility Functions ---


def parse_slack_mention(event_body: dict, bot_user_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a Slack app_mention event body to extract sender ID, channel ID, and cleaned message text.
    
    Args:
        event_body (dict): The raw event payload dictionary from Slack.
        bot_user_id (str): The bot's user ID (e.g., "U0123ABCDE").
        
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing:
            - sender_user_id: The ID of the user who sent the message.
            - channel_id: The ID of the channel where the message was sent.
            - cleaned_text: The message text with the bot mention removed.
    """
    try:
        event = event_body.get('event', {})
        sender_user_id = event.get('user')
        channel_id = event.get('channel')
        raw_text = event.get('text', '')
        
        mention_string = f"<@{bot_user_id}>"
        cleaned_text = raw_text.replace(mention_string, '').strip()
        
        return sender_user_id, channel_id, cleaned_text
        
    except Exception as e:
        logger.error(f"Error in parse_slack_mention: {e}", exc_info=True)
        return None, None, None

# --- Slack App Initialization ---

# Initializes your app with your bot token and signing secret
try:
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
    slack_app_token = os.environ.get("SLACK_APP_TOKEN")

    if not all([slack_bot_token, slack_signing_secret, slack_app_token]):
        missing_tokens = []
        if not slack_bot_token: missing_tokens.append("SLACK_BOT_TOKEN")
        if not slack_signing_secret: missing_tokens.append("SLACK_SIGNING_SECRET")
        if not slack_app_token: missing_tokens.append("SLACK_APP_TOKEN")
        raise ValueError(f"CRITICAL: Missing Slack environment variables: {', '.join(missing_tokens)}")

    app = App(
        token=slack_bot_token,
        signing_secret=slack_signing_secret
    )
    logger.info("Slack App initialized successfully.")
except ValueError as e:
    logger.critical(e)
    exit(1)
except Exception as e:
    logger.critical(f"CRITICAL: Error initializing Slack App: {e}")
    exit(1)


# --- Slack Event Handlers ---

@app.event("app_mention")
def handle_app_mention_events(body: dict, say: callable, logger: logging.Logger):
    # ... (existing bot_user_id extraction and call to parse_slack_mention) ...
    # sender_id, channel_id, user_question = parse_slack_mention(body, bot_user_id)
    # ... (existing handling for empty user_question) ...

    event_ts = body["event"].get("ts", "unknown_ts")
    thread_ts = body["event"].get("thread_ts", event_ts)

    try:
        authorizations = body.get('authorizations')
        if not authorizations or not isinstance(authorizations, list) or not authorizations[0]:
            raise KeyError("Missing or invalid 'authorizations' in event body.")
        bot_user_id = authorizations[0].get('user_id')
        if not bot_user_id:
            raise KeyError("Missing 'user_id' in authorizations.")
    except KeyError as e:
        logger.error(f"Could not retrieve bot_user_id from event body (Event TS: {event_ts}): {e}", exc_info=True)
        say(text="Sorry, I had an internal issue identifying myself. Please try again.", thread_ts=thread_ts)
        return

    sender_id, channel_id, user_question = parse_slack_mention(body, bot_user_id)

    if not sender_id or not channel_id :
        logger.error(f"Could not parse sender_id or channel_id. Event TS: {event_ts}")
        say(text="Sorry, I couldn't understand the details of your message.", thread_ts=thread_ts)
        return

    if not user_question:
        logger.info(f"User <{sender_id}> mentioned bot in channel <{channel_id}> without a question. Event TS: {event_ts}")
        say(text=f"Hi <@{sender_id}>! You mentioned me. Did you have a question?", thread_ts=thread_ts)
        return

    logger.info(f"User <{sender_id}> in channel <{channel_id}> (Thread: {thread_ts}, Event TS: {event_ts}) asked: '{user_question}'")

    # --- NEW: Search Slack History ---
    slack_history_results = search_slack_history(user_question, top_n=3)
    history_context_for_gemini = ""
    history_links_for_user = ""

    if slack_history_results:
        history_links_for_user += "\n\nI found some potentially relevant past discussions:\n"
        for i, result in enumerate(slack_history_results):
            history_links_for_user += f"- <{result.get('permalink', '#')}|View Thread ({result.get('id', 'msg '+str(i+1))})>\n"
            # Prepare context for Gemini (optional, more advanced RAG)
            history_context_for_gemini += f"Relevant past discussion snippet {i+1}:\nUser {result.get('metadata',{}).get('user','Unknown')} said: {result.get('text','')}\nPermalink: {result.get('permalink','')}\n---\n"
        logger.info(f"Found {len(slack_history_results)} results from Slack history search.")
    else:
        logger.info("No relevant results from Slack history search.")
    # --- END NEW ---

    try:
        say(text=f"Oh, hello there <@{sender_id}>! I'm absolutely thrilled to help you with your question! While I'm still learning and developing my capabilities, I'm just overjoyed to be of service! Let me think about that for you... 🤖✨", thread_ts=thread_ts)
        
        # Construct prompt for Gemini
        # Option 1: Simple prompt (Gemini uses its general knowledge)
        # prompt = user_question

        # Option 2: Prompt with retrieved Slack history context (RAG)
        if history_context_for_gemini:
            prompt = (
                f"You are a helpful assistant. Answer the user's question based on your general knowledge "
                f"and the following potentially relevant snippets from past Slack discussions. "
                f"If you use information from the snippets, mention it. If the snippets don't help, answer from your general knowledge.\n\n"
                f"CONTEXT FROM PAST SLACK DISCUSSIONS:\n{history_context_for_gemini}\n\n"
                f"USER QUESTION: {user_question}\n\n"
                f"ANSWER:"
            )
        else:
            prompt = ( # Fallback if no history found, or if you prefer to always add system instruction
                f"You are a helpful assistant. Please answer the following question:\n\n"
                f"USER QUESTION: {user_question}\n\n"
                f"ANSWER:"
            )

        logger.info(f"Sending prompt to Gemini (length: {len(prompt)}): {prompt[:500]}...") # Log start of prompt
        response = model.generate_content(prompt)
        
        gemini_answer = ""
        # ... (rest of your Gemini response handling: safety blocks, empty response, etc.)
        if response.parts:
            gemini_answer = response.text
        elif response.candidates and response.candidates[0].finish_reason == 'SAFETY':
            gemini_answer = "I'm sorry, I can't provide a response to that due to safety guidelines. Please try rephrasing your question or ask something else."
            logger.warning(f"Gemini response for '{prompt[:100]}...' (User: {sender_id}, Event TS: {event_ts}) was blocked due to safety. Response: {response}")
        else:
            gemini_answer = "I received a response, but it didn't contain any text content. Could you try rephrasing?"
            logger.warning(f"Gemini response for '{prompt[:100]}...' (User: {sender_id}, Event TS: {event_ts}) was empty or unexpected. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}. Full response: {response}")
        
        final_response_text = f"<@{sender_id}>, regarding '{user_question}':\n\n{gemini_answer}"
        
        # Append Slack history links if any were found
        if history_links_for_user:
            final_response_text += history_links_for_user
        
        say(text=final_response_text, thread_ts=thread_ts)

    except Exception as e:
        logger.error(f"Error calling Gemini API or processing its response for '{user_question}' (User: {sender_id}, Event TS: {event_ts}): {e}", exc_info=True)
        say(text=f"Sorry <@{sender_id}>, I encountered an issue while trying to answer your question. Please try again in a bit.", thread_ts=thread_ts)

# --- Start Application ---

if __name__ == "__main__":
    logger.info("Starting bot application...")
    try:
        handler = SocketModeHandler(app, slack_app_token) # Use the validated slack_app_token
        logger.info("SocketModeHandler initialized. Bot is attempting to connect and run...")
        handler.start()
    except NameError as e: # Catch if slack_app_token wasn't defined due to earlier exit
        logger.critical(f"Failed to start SocketModeHandler due to NameError (likely missing Slack env vars): {e}")
    except Exception as e:
        logger.critical(f"Failed to start SocketModeHandler: {e}", exc_info=True)
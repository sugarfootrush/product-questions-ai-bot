# slack_history_loader.py

import os
import logging
import time
import json
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Optional, List, Dict, Any # Using List, Dict, Any for more specific typing

# Load environment variables - important if running this script standalone
# Ensure your .env file has SLACK_BOT_TOKEN
load_dotenv()

# Configure logging for this script
# This setup is specific to this script. Your bot.py can have its own.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs logs to the console
        # You could also add logging.FileHandler("slack_loader.log") here
    ]
)
logger = logging.getLogger(__name__)

# Initialize Slack WebClient
try:
    slack_bot_token_for_api = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_bot_token_for_api:
        # Log and raise a specific error if the token is missing
        logger.critical("CRITICAL: SLACK_BOT_TOKEN (for WebClient) not found in .env file or environment variables.")
        raise ValueError("CRITICAL: SLACK_BOT_TOKEN (for WebClient) not found. Please set it in your .env file.")
    slack_client = WebClient(token=slack_bot_token_for_api)
    logger.info("Slack WebClient initialized successfully for API calls.")
except ValueError as e:
    logger.critical(e)
    exit(1) # Exit if critical config like token is missing
except Exception as e: # Catch any other unexpected errors during client initialization
    logger.critical(f"CRITICAL: Unexpected error initializing Slack WebClient: {e}", exc_info=True)
    exit(1)

def get_message_permalink(channel_id: str, message_ts: str) -> Optional[str]:
    """
    Gets a permalink for a given message using the globally defined slack_client.
    """
    global logger, slack_client # Explicitly state usage of global client for clarity if preferred
    try:
        response = slack_client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
        if response.get("ok"):
            return response.get("permalink")
        else:
            logger.warning(f"Failed to get permalink for msg {message_ts} in channel {channel_id}. Error: {response.get('error')}")
    except SlackApiError as e:
        logger.error(f"Slack API Error getting permalink for message {message_ts} in channel {channel_id}: {e.response['error']}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in get_message_permalink for {message_ts}: {e}", exc_info=True)
    return None

def fetch_channel_history(channel_id: str, limit: int = 1000, oldest_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches message history from a specific Slack channel using the globally defined slack_client.
    Includes permalinks for each message and respects rate limits.
    """
    global logger, slack_client # Explicitly state usage of global client for clarity if preferred
    
    messages_data: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    fetched_count: int = 0
    api_calls_history: int = 0 # Counter for conversations.history calls
    api_calls_permalink: int = 0 # Counter for getPermalink calls

    logger.info(f"Attempting to fetch up to {limit} messages from channel '{channel_id}'...")
    try:
        while fetched_count < limit:
            api_calls_history += 1
            logger.debug(
                f"History API Call #{api_calls_history}: Fetching messages for channel '{channel_id}', "
                f"cursor: {cursor}, current fetched count: {fetched_count}"
            )
            response = slack_client.conversations_history(
                channel=channel_id,
                limit=min(200, limit - fetched_count), # Fetch in batches (Slack API limit often 200)
                cursor=cursor,
                oldest=oldest_timestamp # Unix timestamp for 'oldest message to fetch'
            )
            
            if not response.get("ok"):
                logger.error(f"Error from conversations.history for '{channel_id}': {response.get('error')}")
                break

            fetched_messages_batch: List[Dict[str, Any]] = response.get("messages", [])
            if not fetched_messages_batch:
                logger.info(f"No more messages found in channel '{channel_id}' with current parameters.")
                break

            logger.info(f"Retrieved a batch of {len(fetched_messages_batch)} messages.")

            for i, msg in enumerate(fetched_messages_batch):
                # Filter for actual user messages; you might adjust this based on your needs
                if msg.get("type") == "message" and not msg.get("subtype") and msg.get("ts") and msg.get("user"):
                    logger.debug(f"Processing message {i+1}/{len(fetched_messages_batch)} in batch: TS={msg['ts']}")
                    
                    # Get permalink for each message
                    api_calls_permalink += 1
                    permalink = get_message_permalink(channel_id, msg["ts"]) # This makes an API call
                    
                    messages_data.append({
                        "ts": msg["ts"], # Timestamp, also serves as a unique ID
                        "thread_ts": msg.get("thread_ts"), # Important for identifying threads
                        "user": msg.get("user"),
                        "text": msg.get("text", ""), # Ensure text exists, default to empty string
                        "permalink": permalink or "" # Store permalink, default to empty string if None
                    })
                    fetched_count += 1
                    
                    # Delay after each permalink call to be respectful of rate limits
                    # chat.getPermalink is Tier 3 (~50+/min). A 1.2s sleep aims for ~50/min.
                    # You can adjust this; 0.5s was causing issues, so 1.2s is safer.
                    logger.debug(f"Permalink API Call #{api_calls_permalink}. Sleeping for 2.0 seconds...")
                    time.sleep(2.0) 

                    if fetched_count >= limit:
                        logger.info(f"Reached requested message limit of {limit}.")
                        break 
                else:
                    logger.debug(f"Skipping message {i+1}/{len(fetched_messages_batch)} in batch (not a processable user message): TS={msg.get('ts')}")
            
            if fetched_count >= limit: # Check again if inner loop broke due to limit
                break

            # Prepare for next iteration if needed
            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                logger.info(f"Reached the end of messages for channel '{channel_id}' (no next_cursor).")
                break
            
            # IMPORTANT: Respect Slack's rate limits for conversations.history (Tier 3)
            logger.debug(
                f"Sleeping for 1.2 seconds before next conversations.history batch fetch for channel '{channel_id}'."
            )
            time.sleep(2.0) 

    except SlackApiError as e:
        logger.error(f"Slack API Error during channel history fetch for '{channel_id}': {e.response['error']}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during channel history fetch for '{channel_id}': {e}", exc_info=True)
        
    logger.info(
        f"Fetched a total of {len(messages_data)} messages from channel '{channel_id}' "
        f"using {api_calls_history} conversations.history calls and {api_calls_permalink} chat.getPermalink calls."
    )
    return messages_data

if __name__ == "__main__":
    logger.info("Starting script to fetch historical Slack messages from a specific channel.")

    # --- Configuration ---
    # IMPORTANT: Replace this with the actual ID of the channel you want to process!
    # You can get a channel ID from its URL in Slack (e.g., C0123ABCDEF)
    # or by right-clicking the channel -> View channel details -> Copy ID (if your Slack client shows it)
    # or programmatically via other Slack APIs if needed.
    TARGET_CHANNEL_ID = "CHAHQM1P0" 
    MESSAGES_TO_FETCH_LIMIT = 75 # For MVP, fetch a manageable number. Increase as needed.

    if TARGET_CHANNEL_ID == "YOUR_ACTUAL_CHANNEL_ID_HERE":
        logger.error("CRITICAL: Please update TARGET_CHANNEL_ID in slack_history_loader.py before running!")
        exit(1) # Exit if the placeholder ID hasn't been changed.
        
    historical_messages = fetch_channel_history(TARGET_CHANNEL_ID, limit=MESSAGES_TO_FETCH_LIMIT)

    if historical_messages:
        logger.info(f"Successfully fetched {len(historical_messages)} messages from channel '{TARGET_CHANNEL_ID}'.")
        
        # Example: Print some details of the first few fetched messages
        logger.info(f"--- First 5 (max) fetched messages from '{TARGET_CHANNEL_ID}' ---")
        for i, msg_data in enumerate(historical_messages[:5]):
            logger.info(
                f"Msg {i+1}: TS='{msg_data.get('ts')}', User='{msg_data.get('user')}', "
                f"Text='{msg_data.get('text', '')[:60].replace('\n', ' ')}...', " # Show first 60 chars, newline replaced
                f"Permalink='{msg_data.get('permalink', 'N/A')}'"
            )
        logger.info("--- End of sample messages ---")
        
        # TODO: Next step will be to save these messages or send them to your vector DB.
        # For example, to save to a JSON file:
        import json
        import time
        # Save the fetched messages to a JSON file
        output_filename = f'slack_history_{TARGET_CHANNEL_ID}_{time.strftime("%Y%m%d-%H%M%S")}.json'
        try:
             with open(output_filename, 'w', encoding='utf-8') as f:
                 json.dump(historical_messages, f, indent=2, ensure_ascii=False)
             logger.info(f"Saved fetched messages to '{output_filename}'")
        except IOError as e:
             logger.error(f"Error saving messages to JSON file: {e}", exc_info=True)
            
    else:
        logger.warning(f"No messages were fetched from channel '{TARGET_CHANNEL_ID}'. Check channel ID and bot permissions (needs to be in the channel, and have 'channels:history' scope).")

    logger.info("Slack history fetching script finished.")
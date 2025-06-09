# populate_vector_db.py (New File)

import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# --- Configuration ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini (primarily for embedding model)
try:
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("CRITICAL: GOOGLE_API_KEY not found.")
    genai.configure(api_key=gemini_api_key)
    logger.info("Gemini API configured for embeddings.")
except ValueError as e:
    logger.critical(e)
    exit(1)

# Embedding model name (check Google's documentation for the latest/recommended)
EMBEDDING_MODEL_NAME = "models/embedding-001" # Or specific Gemini embedding models

# ChromaDB setup
CHROMA_DATA_PATH = "chroma_data" # Directory to store ChromaDB data
COLLECTION_NAME = "slack_history" # Name of your collection in ChromaDB

# Input JSON file (from slack_history_loader.py)
# Make sure to use the actual filename you generated
INPUT_JSON_FILE = "slack_history_C08T9PM9UAK_20250528-065142.json" # <--- UPDATE THIS FILENAME

def main():
    logger.info(f"Starting script to populate Vector DB from '{INPUT_JSON_FILE}'")

    # 1. Load Slack Messages from JSON
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            slack_messages = json.load(f)
        logger.info(f"Successfully loaded {len(slack_messages)} messages from JSON.")
    except FileNotFoundError:
        logger.error(f"Error: Input JSON file '{INPUT_JSON_FILE}' not found. Please run slack_history_loader.py first.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from '{INPUT_JSON_FILE}'. Ensure it's a valid JSON.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading messages: {e}", exc_info=True)
        return

    if not slack_messages:
        logger.warning("No messages found in the JSON file to process.")
        return

    # 2. Initialize ChromaDB Client and Collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH) # Persists data to disk
        # client = chromadb.Client() # In-memory client, data lost on script exit
        
        # Get or create the collection
        # You can also specify the embedding function here if using ChromaDB's defaults
        # with specific models, but we'll generate embeddings with Gemini explicitly.
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"ChromaDB client initialized and collection '{COLLECTION_NAME}' accessed/created.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
        return

    # 3. Prepare data for ChromaDB (Batching is good for large datasets)
    documents_to_embed = [] # List of texts
    metadatas_to_store = [] # List of corresponding metadata dicts
    ids_for_documents = []  # List of unique IDs

    for i, msg in enumerate(slack_messages):
        text_content = msg.get("text")
        doc_id = msg.get("ts") # Using timestamp as a unique ID

        if not text_content or not doc_id:
            logger.warning(f"Skipping message {i+1} due to missing text or ts: {msg}")
            continue

    # Ensure all metadata fields are explicitly strings or valid primitives,
    # defaulting None values to empty strings or appropriate placeholders.
    if text_content.strip(): 
        documents_to_embed.append(text_content)

        # Construct metadata, ensuring no None values are passed for string fields
        current_metadata = {
            "permalink": str(msg.get("permalink")) if msg.get("permalink") is not None else "",
            "user": str(msg.get("user")) if msg.get("user") is not None else "",
            "thread_ts": str(msg.get("thread_ts")) if msg.get("thread_ts") is not None else "",
            "original_ts": str(doc_id) # doc_id is confirmed not None here, ensure it's a string
        }
        metadatas_to_store.append(current_metadata)
        ids_for_documents.append(str(doc_id)) # Ensure ID is also a string
        
        # For an MVP, we only embed non-empty text.
        # You might add more sophisticated filtering/preprocessing later

    if not documents_to_embed:
        logger.warning("No valid documents to embed after filtering.")
        return
        
    logger.info(f"Prepared {len(documents_to_embed)} documents for embedding.")

    # 4. Generate Embeddings (Batching if supported by Gemini API for many docs, or loop)
    # Gemini's embed_content often handles one piece of content at a time for 'CONTENT' role.
    # For batching, the API might be different or you'd loop.
    # Let's do it one by one for clarity, but batching is more efficient for many.
    
    embeddings_list = []
    logger.info(f"Generating embeddings using '{EMBEDDING_MODEL_NAME}'...")
    for i, doc_text in enumerate(documents_to_embed):
        try:
            # Ensure content is not empty before sending to API
            if not doc_text.strip():
                logger.warning(f"Skipping embedding for empty document at index {i}.")
                # We need a placeholder or to skip this doc in ids and metadatas too if we do this.
                # For simplicity, earlier filtering should prevent this.
                continue

            # Using task_type "RETRIEVAL_DOCUMENT" is often recommended for documents to be stored for later retrieval.
            # For questions, you'd use "RETRIEVAL_QUERY".
            embedding_result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=doc_text,
                task_type="RETRIEVAL_DOCUMENT" # Or "SEMANTIC_SIMILARITY" or other relevant task types
            )
            embeddings_list.append(embedding_result['embedding'])
            if (i + 1) % 10 == 0: # Log progress every 10 embeddings
                logger.info(f"Generated {i+1}/{len(documents_to_embed)} embeddings...")
        except Exception as e:
            logger.error(f"Error generating embedding for document: '{doc_text[:50]}...': {e}", exc_info=True)
            # Decide how to handle: skip this doc, add a None embedding (if Chroma allows), or stop.
            # For now, we'll just log and continue, meaning this doc won't be added.
            # This would cause a mismatch if not handled carefully when adding to Chroma.
            # A robust solution would be to remove corresponding items from ids_for_documents and metadatas_to_store.
            # For simplicity here, we'll assume successful embedding for all valid docs.
            # (This part needs more robust error handling for production)

    if len(embeddings_list) != len(ids_for_documents):
        logger.error("Mismatch between number of embeddings and documents due to errors. Aborting ChromaDB update.")
        # This is a placeholder for more robust error handling logic.
        # You'd need to align ids_for_documents and metadatas_to_store with successfully generated embeddings.
        return

    logger.info(f"Successfully generated {len(embeddings_list)} embeddings.")

    # 5. Add to ChromaDB
    if embeddings_list:
        try:
            logger.info(f"Adding {len(embeddings_list)} embeddings to ChromaDB collection '{COLLECTION_NAME}'...")
            collection.add(
                embeddings=embeddings_list,
                documents=documents_to_embed, # Storing the original text alongside embedding
                metadatas=metadatas_to_store,
                ids=ids_for_documents
            )
            logger.info("Successfully added data to ChromaDB.")
            logger.info(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} items.")
        except Exception as e:
            logger.error(f"Error adding data to ChromaDB: {e}", exc_info=True)
    else:
        logger.warning("No embeddings were generated to add to ChromaDB.")

if __name__ == "__main__":
    main()
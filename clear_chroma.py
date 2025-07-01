import chromadb

CHROMA_DATA_PATH = r"C:\Users\MercerMike\Projects\product-questions-ai-bot\chroma_data"
COLLECTION_NAME = "slack_history" # Update if your collection name is different

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Get all IDs in the collection
all_ids = []
# ChromaDB stores ids in batches, so you may need to use get() with a large limit or iterate
# Here, we use get() with a very large limit to try to fetch all
results = collection.get(limit=1000000)
if "ids" in results:
    all_ids = results["ids"]

if all_ids:
    collection.delete(ids=all_ids)
    print(f"Deleted {len(all_ids)} documents from the collection.")
else:
    print("No documents found in the collection to delete.")

print("ChromaDB collection cleared!") 
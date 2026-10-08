import chromadb

client = chromadb.PersistentClient(path="chroma_db")

print("Available collections:")

for collection in client.list_collections():
    print("\nCollection:", collection.name)

    data = client.get_collection(collection.name)
    print("Document count:", data.count())

    if data.count() > 0:
        results = data.get(limit=3, include=["documents"])

        print("Sample documents:")
        for doc in results["documents"]:
            print("--------------------")
            print(doc[:500])
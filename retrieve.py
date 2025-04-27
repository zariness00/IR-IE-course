import chromadb
main_path = "/Users/zoryawka/Desktop/Coding/IR-IE-course/advanced"
COLLECTION_NAME  = "advanced"

def main():
    client     = chromadb.PersistentClient(path= main_path)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    result = collection.get("Please Please Please_0")
    print(result)

    result = collection.get(where={"chunk_index": { "$eq":11}})
    print(result)



if __name__ == "__main__":
    main()

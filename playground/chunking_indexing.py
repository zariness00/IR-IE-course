import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

csv_file_path = "/Users/zoryawka/Desktop/Coding/IR-IE-course/lyrics/sabrina_carpenter_cleaned.csv"

df = pd.read_csv(csv_file_path)

song_splitter = RecursiveCharacterTextSplitter(
    separators=[". ", "? ", "! ", "\n", "\r\n"],
    chunk_size=500,
    chunk_overlap=20,
)


chunks_sabrina = []
for _, row in df.iterrows():
    song_id = str(row["index"])
    song_title = str(row["Song"])
    song_lyrics = str(row["Lyrics"])
    docs = song_splitter.create_documents([song_lyrics])
    for i, doc in enumerate(docs, start=1):
        doc.metadata.update({
            "song_id": song_id,
            "song_title": song_title, 
            "chunk_index": i # number of the chunk within the song 
        })
        chunks_sabrina.append(doc)

#checking the contents of the chunks
# n_chunk = chunks_sabrina[4]
# print("Sabrina Carpenter - First Chunk Content:")
# print(n_chunk.page_content)
# print("Metadata:", n_chunk.metadata)


client_chroma = chromadb.PersistentClient(path="./advanced")
collection = client_chroma.get_or_create_collection(name = "advanced", metadata = {"hnsw:space": "cosine"})
#verifying my collection
#print(f"ChromaDB collection {client_chroma.list_collections()}")


for idx, chunk in enumerate(chunks_sabrina):
    text = chunk.page_content
    chunk.metadata["chunk_index"] = idx
    collection.add(
        documents=[text],
        ids=[f"{chunk.metadata['song_title']}_{idx}"],
        metadatas=[chunk.metadata]
    )

print(f"Indexed {len(chunks_sabrina)} chunks into collection 'advanced'.")


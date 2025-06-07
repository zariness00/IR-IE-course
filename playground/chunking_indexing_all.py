import os
import glob
import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv


load_dotenv("playground/genius_api.env")
csv_folder = os.getenv("CSV_FOLDER")

CHROMA_PATH      = "./my_collection_1"
COLLECTION_NAME  = "my_collection_1"


def split_lyrics_from_csv(csv_file_path):
    """
    Reads a CSV that must contain at least these columns:
       - Artist   (string)
       - Title    (string)
       - Album    (string)        [optional: if missing, default to empty str]
       - Year     (int or str)    [optional: if missing, default to None]
       - Lyric    (string)
    Splits each row’s full lyric into ~500‐char chunks (20‐char overlap).
    Returns a list of LangChain Document objects, with metadata:
      artist, song_title, song_album, song_year, chunk_index.
    """
    df = pd.read_csv(csv_file_path)
    splitter = RecursiveCharacterTextSplitter(
        separators=[". ", "? ", "! ", "\n", "\r\n", "  ", "/n/n"],
        chunk_size=500,
        chunk_overlap=20,
    )
    chunks = []
    for _, row in df.iterrows():
        artist     = str(row.get("Artist", "")).strip()
        song_title = str(row.get("Title", "")).strip()
        song_album = str(row.get("Album", "")).strip()
        try:
            song_year = int(row["Year"])
        except Exception:
            song_year = ""
        song_lyrics = str(row.get("Lyric", "")).strip()
        if not song_lyrics:
            continue  # skip rows with no lyric text

        # split this one song’s lyric text into smaller chunks
        docs_for_this_song = splitter.create_documents([song_lyrics])

        # annotate each chunk with metadata & append to my list
        for idx, doc in enumerate(docs_for_this_song, start=1):
            doc.metadata.update({
                "artist":      artist,
                "song_title":  song_title,
                "song_album":  song_album,
                "song_year":   song_year,
                "chunk_index": idx
            })
            chunks.append(doc)
    return chunks


if __name__ == "__main__":
    all_csv_paths = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not all_csv_paths:
        print(f"No CSVs found in {csv_folder}. Exiting.")
        exit(0)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})
    except Exception:
        collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})
    total_indexed = 0


    #  Loop over each CSV and ingest its chunks
    for csv_path in all_csv_paths:
        print(f"\nProcessing {os.path.basename(csv_path)} …")
        try:
            chunks = split_lyrics_from_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        if not chunks:
            print("(No valid lyric chunks found—skipping.)")
            continue

        # Build parallel lists of documents, ids, metadatas
        docs      = [chunk.page_content for chunk in chunks]
        ids       = [f"{chunk.metadata['artist'].replace(' ', '_')}_{i}"
                    for i, chunk in enumerate(chunks, start=1)]
        metadatas = [chunk.metadata for chunk in chunks]
        collection.add(
            documents=docs,
            ids=ids,
            metadatas=metadatas
        )
        total_indexed += len(chunks)
        print(f"Indexed {len(chunks)} chunks from {os.path.basename(csv_path)}")
 
    print(f"CHECKPOINT: Total chunks indexed (all artists): {total_indexed}")
    print(f"CHECKPOINT: ChromaDB collection stored at {CHROMA_PATH}, name = {COLLECTION_NAME}")


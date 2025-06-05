import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

main_path = "/Users/zoryawka/Desktop/Coding/IR-IE-course/final_collection"
COLLECTION_NAME  = "final_collection"

load_dotenv("api_key.env")
api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key) 

client     = chromadb.PersistentClient(path= main_path)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
def get_completion(prompt):
    response = client_openai.chat.completions.create(
        model= "gpt-4",
        messages=[
            {"role":"system", "content": "You're a helpful assistant who retrieves information from external sources and presents them to the user."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content
# def main():
#     client     = chromadb.PersistentClient(path= main_path)
#     collection = client.get_or_create_collection(name=COLLECTION_NAME)
#     result = collection.get("Please Please Please_0")
#     print(result)

#     result = collection.get(where={"chunk_index": { "$eq":11}})
#     print(result)

def populate_rag_query(query, n_results=1):
    client     = chromadb.PersistentClient(path= main_path)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for idx, result in enumerate(search_results["documents"][0]):
        metadata = search_results["metadatas"][0][idx]
        formatted_result = f"""<SEARCH RESULT>
                                <DOCUMENT>{result}</DOCUMENT>
                                <METADATA>
                                    <ARTIST>{metadata.get('artist', '')}</ARTIST>
                                    <TITLE>{metadata.get('song_title', '')}</TITLE>
                                    <ALBUM>{metadata.get('song_album', '')}</ALBUM>
                                    <YEAR>{metadata.get('song_year', '')}</YEAR>
                                    <CHUNK_IDX>{metadata.get('chunk_index', '')}</CHUNK_IDX>
                                </METADATA>
                                </SEARCH RESULT>
                                """
        result_str += formatted_result
    return result_str

def make_rag_prompt(query, results):
    return f"""<INSTRUCTIONS>
   <EXAMPLE CITATION>
   Answer to the user query in your own words, drawn from the search results.
   - "Direct quote from source material backing up the claim" - [Source: Song Title, Artist, Album, Year]
   </EXAMPLE CITATION>
   </INSTRUCTIONS>

    <USER QUERY>
    {query}
    </USER QUERY>

    <SEARCH RESULTS>
    {results}
    </SEARCH RESULTS>

    Your answer:"""
# all_data = collection.get()                   # returns all documents
# all_ids  = all_data["ids"]                    # list of every ID in the collection
# print(f"Total IDs: {len(all_ids)}")
# print(all_ids[1900:1967])
# #print(collection.get('Billie_Eilish_10'))
# # print(collection.get('Maroon_5_0'))
# print(collection.get('Justin_Bieber_1'))

# ─── 1) Helper to fetch a single chunk dict by its chunk_index ────────────
def fetch_chunk_by_index(chunk_idx):
    """
    Returns a single‐chunk dict from ChromaDB where chunk_index == chunk_idx.
    If no chunk is found, returns None.

    This version handles both “nested‐list” and “flat‐list” return shapes
    that ChromaDB might produce.
    """
    result = collection.get(where={"chunk_index": {"$eq": chunk_idx}})
    if not result:
        return None

    # --- 1) Extract a single ID ---
    ids_outer = result.get("ids")
    if not ids_outer:
        return None
    first_id_entry = ids_outer[0]
    if isinstance(first_id_entry, list):
        if not first_id_entry:
            return None
        single_id = first_id_entry[0]
    else:
        single_id = first_id_entry

    # --- 2) Extract a single document string ---
    docs_outer = result.get("documents")
    if not docs_outer:
        return None
    first_doc_entry = docs_outer[0]
    if isinstance(first_doc_entry, list):
        if not first_doc_entry:
            return None
        single_doc = first_doc_entry[0]
    else:
        single_doc = first_doc_entry

    # --- 3) Extract a single metadata dict ---
    metas_outer = result.get("metadatas")
    if not metas_outer:
        return None
    first_meta_entry = metas_outer[0]
    if isinstance(first_meta_entry, list):
        if not first_meta_entry:
            return None
        single_meta = first_meta_entry[0]
    else:
        single_meta = first_meta_entry

    # Return in the exact shape that expanded_search_results (and the rest
    # of your code) expects: keys "ids", "documents", "metadatas" each mapping
    # to a one‐element list.
    return {
        "ids":       [single_id],
        "documents": [single_doc],
        "metadatas": [single_meta],
    }


# ─── 2) Expand around a given chunk dict ────────────────────────────────────
def expanded_search_results(original_chunk):
    """
    original_chunk should be the dict you get from fetch_chunk_by_index(...),
    of the form:
      {
        "id":       [the single ID string],
        "document": [the single chunk text],
        "metadata": [the single metadata dict]
      }
    We pull its chunk_index from metadata, fetch the previous and next chunks,
    and then format all three into <SEARCH RESULT> blocks.
    """
    # Extract the chunk_index of the original
    orig_meta  = original_chunk["metadatas"][0]
    orig_index = orig_meta["chunk_index"]

    # Fetch previous and next chunk dicts (or None if not found)
    prev_chunk = fetch_chunk_by_index(orig_index - 1)
    next_chunk = fetch_chunk_by_index(orig_index + 1)

    result_str = ""

    # Loop over previous, original, next (skip None)
    for chunk_dict in (prev_chunk, original_chunk, next_chunk):
        if chunk_dict is None:
            continue

        # chunk_dict["document"] is a list with one string; likewise for metadata
        doc_text = chunk_dict["documents"][0]
        meta     = chunk_dict["metadatas"][0]

        formatted = f"""<SEARCH RESULT>
  <DOCUMENT>{doc_text}</DOCUMENT>
  <METADATA>
    <ARTIST>{meta.get("artist", "")}</ARTIST>
    <TITLE>{meta.get("song_title", "")}</TITLE>
    <ALBUM>{meta.get("song_album", "")}</ALBUM>
    <YEAR>{meta.get("song_year", "")}</YEAR>
    <CHUNK_IDX>{meta.get("chunk_index", "")}</CHUNK_IDX>
  </METADATA>
</SEARCH RESULT>
"""
        result_str += formatted

    return result_str



def make_decoupled_rag_prompt(query, n_results=3):
    # Run the ChromaDB semantic query
    search_results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    total_result_str = ""

    # Loop over the *first* inner list (index 0), which holds your n_results items
    for doc_text, metadata in zip(
        search_results["documents"][0],
        search_results["metadatas"][0]
    ):
        # Wrap each result into the format expected by expanded_search_results()
        chunk = {
            "documents": [doc_text],
            "metadatas": [metadata]
        }

        # Expand to prev + current + next chunk
        expanded_result = expanded_search_results(chunk)
        total_result_str += expanded_result

    # After gathering all expanded results, build the final RAG prompt
    rag_prompt = make_rag_prompt(query, total_result_str)
    return rag_prompt

prompt_1 = make_decoupled_rag_prompt("songs mentioning war", n_results=3)
rag_completion_1 = get_completion(prompt_1)
print(rag_completion_1)
# prompt_2 = make_decoupled_rag_prompt("songs about making babies", n_results=3)
# rag_completion_2 = get_completion(prompt_2)
# print(rag_completion_2)


# # #this function takes a chunk index of the retrieved search results
# # #then it return the previous and next chunks
# def get_next_and_previous_chunks(chunk_index):
#   previous_chunk = collection.get(where = {"chunk_index": {"$eq": chunk_index - 1}})
#   next_chunk =  collection.get(where = {"chunk_index": {"$eq": chunk_index + 1}})
#  #passing chunk_idx to the where argument

#   return previous_chunk, next_chunk


# #this function accepts the original chink and returns a string of search reuslts for the previous, current, and next chunks
# # def expanded_search_results(original_chunk):
# #     original_chunk_idx = original_chunk["metadatas"][0]["chunk_index"]
# #     previous_chunk, next_chunk = get_next_and_previous_chunks(original_chunk_idx)
# #     result_str = ""
# #     for chunk in [previous_chunk, original_chunk, next_chunk]:
# #         if len(chunk["metadatas"])>0:
# #             metadata = chunk["metadatas"][0]
# #             formatted_result = f"""<SEARCH RESULT>
# #             <DOCUMENT>{chunk["documents"][0]}</DOCUMENT>
# #             <METADATA>
# #             <TITLE>{metadata["song_title"]}</TITLE>
# #             <CHUNK_IDX>{metadata["chunk_index"]}</CHUNK_IDX>
# #             </METADATA>
# #             </SEARCH RESULT>"""
# #             result_str += formatted_result
# #     return result_str
# # original_demo_chunk = collection.get(where={"chunk_index": {"$eq": 11}})
# # print("Let's see what we have ")
# # expanded_results = expanded_search_results(original_demo_chunk)
# # print(expanded_results) I DONT need this 


# def expanded_search_results(original_chunk):
#     """
#     `original_chunk` is the dict you get from `collection.get(...)`, which looks like:
#       {
#         "ids":       [...],
#         "documents": [...],
#         "metadatas": [...]
#       }
#     We take the first (and only) metadata/document in that result, find its chunk_index,
#     then fetch the previous and next chunks by chunk_index, and format all three.
#     """
#     # 1) Pull out the chunk_index of the “original” chunk
#     original_chunk_idx = original_chunk["metadatas"][1]

#     # 2) Grab the previous and next chunks (these functions should return the same dict shape)
#     previous_chunk, next_chunk = get_next_and_previous_chunks(original_chunk_idx)

#     result_str = ""

#     # 3) Loop over [previous, original, next]
#     for chunk in [previous_chunk, original_chunk, next_chunk]:
#         # chunk is a dict with keys "documents" and "metadatas"
#         if chunk and len(chunk["metadatas"]) > 0:
#             meta = chunk["metadatas"][1]
#             # doc_text = chunk["documents"][0]

#             formatted_result = f"""<SEARCH RESULT>
#   <DOCUMENT>{chunk}</DOCUMENT>
#   <METADATA>
#     <ARTIST>{meta.get("artist", "")}</ARTIST>
#     <TITLE>{meta.get("song_title", "")}</TITLE>
#     <ALBUM>{meta.get("song_album", "")}</ALBUM>
#     <YEAR>{meta.get("song_year", "")}</YEAR>
#     <CHUNK_IDX>{meta.get("chunk_index", "")}</CHUNK_IDX>
#   </METADATA>
# </SEARCH RESULT>
# """
#             result_str += formatted_result

#     return result_str


# # Example usage:
# original_demo_chunk = collection.get(where={"chunk_index": {"$eq": 11}})
# print("Let's see what we have:")
# expanded_results = expanded_search_results(original_demo_chunk)
# print(expanded_results)



# def make_decoupled_rag_prompt(query, n_results=3):

#     search_results = collection.query(query_texts=[query], n_results=n_results)
#     total_result_str = ""
#     for doc, metadata in zip(search_results['documents'][1], search_results['metadatas'][1]):
#         chunk = {
#             'documents': [doc],
#             'metadatas': [metadata]
#         }

#         expanded_result = expanded_search_results(chunk)
#         total_result_str += expanded_result
#         rag_prompt = make_rag_prompt(query, total_result_str)
#     return rag_prompt

# # given a search result, we can get everything back in a formatted string
# # prompt = make_decoupled_rag_prompt("song about a desire to be pregnant")
# # rag_completion = get_completion(prompt)
# # print(rag_completion)


# prompt_1 = make_decoupled_rag_prompt("songs about a painful breakup", n_results=3)
# rag_completion_1 = get_completion(prompt_1)
# print(rag_completion_1)
# # if __name__ == "__main__":
# #     main()

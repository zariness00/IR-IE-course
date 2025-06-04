import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

main_path = "/Users/zoryawka/Desktop/Coding/IR-IE-course/advanced"
COLLECTION_NAME  = "advanced"

load_dotenv(find_dotenv())
api_key = os.getenv(OPENAI_API_KEY)
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
        <TITLE>{metadata['song_title']}</TITLE>
        <ID>{metadata['song_id']}</ID>
        <CHUNK_IDX>{metadata['chunk_index']}</CHUNK_IDX>
        </METADATA>
        </SEARCH RESULT>"""
        result_str += formatted_result
    return result_str

def make_rag_prompt(query, results):
    return f"""<INSTRUCTIONS>
   <EXAMPLE CITATION>
   Answer to the user query in your own words, drawn from the search results.
   - "Direct quote from source material backing up the claim" - [Source: Song Title, Author, Chunk: chunk index]
   </EXAMPLE CITATION>
   </INSTRUCTIONS>

    <USER QUERY>
    {query}
    </USER QUERY>

    <SEARCH RESULTS>
    {results}
    </SEARCH RESULTS>

    Your answer:"""

def get_next_and_previous_chunks(chunk_idx):
  previous_chunk = collection.get(where = {"chunk_index": {"$eq": chunk_idx - 1}})
  next_chunk =  collection.get(where = {"chunk_index": {"$eq": chunk_idx + 1}})
 #passing chunk_idx to the where argument

  return previous_chunk, next_chunk


#this function takes a chunk index of the retrieved search results
#then it return the previous and next chunks
def get_next_and_previous_chunks(chunk_idx):
  previous_chunk = collection.get(where = {"chunk_index": {"$eq": chunk_idx - 1}})
  next_chunk =  collection.get(where = {"chunk_index": {"$eq": chunk_idx + 1}})
 #passing chunk_idx to the where argument

  return previous_chunk, next_chunk


#this function accepts the original chink and returns a string of search reuslts for the previous, current, and next chunks
def expanded_search_results(original_chunk):
    original_chunk_idx = original_chunk["metadatas"][0]["chunk_index"]
    previous_chunk, next_chunk = get_next_and_previous_chunks(original_chunk_idx)
    result_str = ""
    for chunk in [previous_chunk, original_chunk, next_chunk]:
        if len(chunk["metadatas"])>0:
            metadata = chunk["metadatas"][0]
            formatted_result = f"""<SEARCH RESULT>
            <DOCUMENT>{chunk["documents"][0]}</DOCUMENT>
            <METADATA>
            <TITLE>{metadata["song_title"]}</TITLE>
            <CHUNK_IDX>{metadata["chunk_index"]}</CHUNK_IDX>
            </METADATA>
            </SEARCH RESULT>"""
            result_str += formatted_result
    return result_str
original_demo_chunk = collection.get(where={"chunk_index": {"$eq": 11}})
print("Let's see what we have ")
expanded_results = expanded_search_results(original_demo_chunk)
print(expanded_results)


def make_decoupled_rag_prompt(query, n_results=3):

    search_results = collection.query(query_texts=[query], n_results=n_results)
    total_result_str = ""
    for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
        chunk = {
            'documents': [doc],
            'metadatas': [metadata]
        }

        expanded_result = expanded_search_results(chunk)
        total_result_str += expanded_result
        rag_prompt = make_rag_prompt(query, total_result_str)
    return rag_prompt

# given a search result, we can get everything back in a formatted string
# prompt = make_decoupled_rag_prompt("song about a desire to be pregnant")
# rag_completion = get_completion(prompt)
# print(rag_completion)


prompt_1 = make_decoupled_rag_prompt("song about a guy who is dumb")
rag_completion_1 = get_completion(prompt_1)
print(rag_completion_1)
# if __name__ == "__main__":
#     main()

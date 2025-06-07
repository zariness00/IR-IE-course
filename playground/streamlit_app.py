import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import re


st.title("POP RAG Search Engine")


load_dotenv("api_key.env")
api_key = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=api_key)


MAIN_PATH      = os.getenv("COLLECTION_PATH")
COLLECTION_NAME = "my_collection"

client     = chromadb.PersistentClient(path=MAIN_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})

def get_completion(prompt):
    response = client_openai.chat.completions.create(
        model= "gpt-4",
        messages=[
            {"role":"system", "content": "You're a helpful assistant who retrieves information from external sources and presents them to the user. Provide an overview, then list of recommended songs and each song is followed the example citation. Pay attention to the n_results "},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def populate_rag_query(query, n_results=1):
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
        </SEARCH RESULT>"""
        result_str += formatted_result
    return result_str

def make_rag_prompt(query, results):
    return f"""<INSTRUCTIONS>
    <EXAMPLE CITATION>
    Answer to the user query in your own words, drawn from the search results. 
    AND - "Direct quote from source material backing up the claim" - [Source: Song Title, Artist, Album, Year]
    </EXAMPLE CITATION>
    When you finish outlining, please output a **numbered list** of the recommended songs
    </INSTRUCTIONS>
    <USER QUERY>
    {query}
    </USER QUERY>
    <SEARCH RESULTS>
    {results}
    </SEARCH RESULTS>

    Your answer:"""
def get_prev_next_chunks(chunk_index: int):
    
    prev_chunk = collection.get(where={"chunk_index": {"$eq": chunk_index - 1}})
    next_chunk = collection.get(where={"chunk_index": {"$eq": chunk_index + 1}})
    return prev_chunk, next_chunk

def expanded_search_results(original_chunk):
    original_chunk_idx = original_chunk["metadatas"][0]["chunk_index"]
    prev_chunk, next_chunk = get_prev_next_chunks(original_chunk_idx)
    result_str = ""
    for chunk in [prev_chunk, original_chunk, next_chunk]:
        if len(chunk["metadatas"])>0:
            meta= chunk["metadatas"][0]
            formatted_result = f"""<SEARCH RESULT>
            <DOCUMENT>{chunk["documents"][0]}</DOCUMENT>
            <METADATA>
            <ARTIST>{meta.get("artist", "")}</ARTIST>
            <TITLE>{meta.get("song_title", "")}</TITLE>
            <ALBUM>{meta.get("song_album", "")}</ALBUM>
            <YEAR>{meta.get("song_year", "")}</YEAR>
            <CHUNK_IDX>{meta.get("chunk_index", "")}</CHUNK_IDX>
            </METADATA>
            </SEARCH RESULT>"""
            result_str += formatted_result
    return result_str

def make_decoupled_rag_prompt(query, n_results=1):
    search_results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents","metadatas"]
        )
    
    total_result_str = ""
    for doc_text, metadata in zip(
        search_results["documents"][0],
        search_results["metadatas"][0]
    ):
        chunk = {
            "documents": [doc_text],
            "metadatas": [metadata]
        }

        expanded_result = expanded_search_results(chunk)
        total_result_str += expanded_result
        rag_prompt = make_rag_prompt(query, total_result_str)

    return rag_prompt





user_question = st.text_input("Enter your query:", placeholder="e.g. songs about a painful breakup")
n_results     = st.slider("Number of suggestions you would like to receive:", min_value=1, max_value=10, value=3)

if st.button("Search and Answer"):
    if not user_question.strip():
        st.warning("Please enter a question before searching.")
    else:
        rag_prompt = make_decoupled_rag_prompt(user_question, n_results)
        rag_answer = get_completion(rag_prompt)
        st.subheader("Answer:")
        st.write(rag_answer)

        # st.subheader("Here are detailed answer:")
        # raw = rag_prompt.split("<SEARCH RESULTS>")[-1].strip()
        # for idx, block in enumerate(raw.split("<SEARCH RESULT>")[1:], start=1):
        #     match = re.search(r"<CHUNK_IDX>(\d+)</CHUNK_IDX>", block)
        #     if match and int(match.group(1)) in central_indices:
        #         st.code("<SEARCH RESULT>" + block)
        #         st.write("---")
        #     # st.markdown(f"**Block {idx}:**")
        #     # st.code("<SEARCH RESULT>" + block)  
        #     # st.write("---")

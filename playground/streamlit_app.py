import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ─── Streamlit App Setup ─────────────────────────────────────────────────────

st.title("Lyrics RAG Playground")

# Load OpenAI API key from .env
load_dotenv("api_key.env")
api_key = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=api_key)

# ChromaDB configuration
MAIN_PATH      = "/Users/zoryawka/Desktop/Coding/IR-IE-course/final_collection"
COLLECTION_NAME = "final_collection"

client     = chromadb.PersistentClient(path=MAIN_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ─── User Inputs ─────────────────────────────────────────────────────────────

user_question = st.text_input("Enter your query:", placeholder="e.g. songs about a painful breakup")
n_results     = st.slider("Number of chunks to retrieve:", min_value=1, max_value=5, value=3)

if st.button("Search and Answer"):
    if not user_question.strip():
        st.warning("Please enter a question before searching.")
    else:
        # ─── 1) Retrieve top‐k chunks from ChromaDB ────────────────────────
        results = collection.query(
            query_texts=[user_question],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        # ─── 2) Format search results for RAG prompt ───────────────────────
        search_results = []
        docs_list  = results["documents"][0]
        metas_list = results["metadatas"][0]

        for doc_text, meta in zip(docs_list, metas_list):
            metadata_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            search_results.append(f"{doc_text}\nMetadata: {metadata_str}")

        search_text = "\n\n".join(search_results)

        # ─── 3) Build RAG prompt ────────────────────────────────────────────
        prompt = f"""Your task is to answer the following user question using the supplied search results.

User Question: {user_question}

Search Results:
{search_text}

Answer in a clear, concise way, citing relevant lyrics snippets."""
        
        # ─── 4) Get completion from OpenAI ─────────────────────────────────
        response = client_openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers using song lyric snippets."},
                {"role": "user",   "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        # ─── 5) Display results ────────────────────────────────────────────
        st.subheader("Retrieved Chunks:")
        for i, entry in enumerate(search_results, start=1):
            st.markdown(f"**Chunk {i}:**")
            st.write(entry)
            st.write("---")

        st.subheader("Answer:")
        st.write(answer)

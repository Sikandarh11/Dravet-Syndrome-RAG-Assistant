# populate_database.py
# Builds a FAISS-backed vectorstore from all PDFs in `data/` folder.
# Usage: python populate_database.py          # to create/update index
#        python populate_database.py --reset  # to clear and rebuild

import argparse
import os
import shutil
from pathlib import Path
import pickle
from config_loader import load_config

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate



from get_embedding_function import get_embedding_function

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config()


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(SCRIPT_DIR, path_value)


# Paths
DATA_PATH = _resolve_path(CONFIG["paths"]["data"])
VECTORSTORE_PATH = _resolve_path(CONFIG["paths"]["vectorstore"])   # directory where FAISS index + docs are saved
DOCS_META_FILE = os.path.join(VECTORSTORE_PATH, "docs.pkl")

CHUNK_SIZE = CONFIG["database"]["chunk_size"]
CHUNK_OVERLAP = CONFIG["database"]["chunk_overlap"]
OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"]
EMBEDDING_MODEL = CONFIG["ollama"]["embedding_model"]

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        print(f"Deleted {VECTORSTORE_PATH}")

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    print(f"Loading PDFs from {DATA_PATH} ...")
    docs = loader.load()
    print(f"Loaded {len(docs)} raw documents (pages).")
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the vectorstore.")
    args = parser.parse_args()
    if args.reset:
        clear_vectorstore()

    # Make sure data folder exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data folder '{DATA_PATH}' not found. Put PDFs into that folder.")
        return

    docs = load_documents()
    chunks = split_documents(docs)
    chunks = calculate_chunk_ids(chunks)

    # Build or update FAISS
    embeddings = get_embedding_function()

    # If vectorstore exists, load and append new docs avoiding duplicates
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCS_META_FILE):
        print("Existing vectorstore found; loading...")
        try:
            db = FAISS.load_local(VECTORSTORE_PATH, embeddings)
            with open(DOCS_META_FILE, "rb") as f:
                existing_docs = pickle.load(f)
        except Exception as e:
            print("Failed to load existing vectorstore (will rebuild):", str(e))
            clear_vectorstore()
            db = None
            existing_docs = []
    else:
        db = None
        existing_docs = []

    # Map existing IDs for dedup
    existing_ids = set()
    for d in existing_docs:
        if isinstance(d, dict):
            existing_ids.add(d.get("metadata", {}).get("id"))
        else:
            existing_ids.add(getattr(d, "metadata", {}).get("id"))

    # Filter new chunks
    new_chunks = [c for c in chunks if c.metadata.get("id") not in existing_ids]

    if not new_chunks and db is not None:
        print("No new chunks to add. Vectorstore up to date.")
        return

    # If no existing db, create from scratch
    if db is None:
        print("Creating new FAISS index from documents...")
        db = FAISS.from_documents(new_chunks, embeddings)
        docs_to_save = new_chunks
    else:
        print(f"Adding {len(new_chunks)} new chunks to existing index...")
        db.add_documents(new_chunks)
        docs_to_save = existing_docs + new_chunks

    # save vectorstore and metadata
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    db.save_local(VECTORSTORE_PATH)
    with open(DOCS_META_FILE, "wb") as f:
        pickle.dump(docs_to_save, f)

    print(f"Done. Vectorstore saved to '{VECTORSTORE_PATH}' with {len(docs_to_save)} documents.")

if __name__ == "__main__":
    main()

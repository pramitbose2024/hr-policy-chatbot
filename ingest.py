import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DOCS_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load all documents from the docs/ folder
# ─────────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list:
    
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"The folder '{docs_dir}' does not exist. "
        )

    supported_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.docx"))

    if not supported_files:
        raise ValueError(
            f"No PDF or DOCX files found in '{docs_dir}'. "
        )

    print(f"\n Found {len(supported_files)} document(s) in '{docs_dir}':")

    for file_path in supported_files:
        print(f"   Loading: {file_path.name}", end=" ... ")
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))

            docs = loader.load()

            # Attach the filename to every page's metadata
            # This is critical — later when the chatbot retrieves a chunk,
            # you can tell the user exactly which document it came from
            for doc in docs:
                doc.metadata["source"] = file_path.name

            documents.extend(docs)
            print(f" {len(docs)} page(s)")

        except Exception as e:
            print(f" Failed — {e}")
            continue

    print(f"\n Total pages loaded: {len(documents)}")
    return documents


# ─────────────────────────────────────────────────────────────
# STEP 2: Split documents into chunks
# ─────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"\n  Splitting complete:")
    print(f"   Pages loaded   : {len(documents)}")
    print(f"   Chunks created : {len(chunks)}")
    print(f"   Chunk size     : {CHUNK_SIZE} characters")
    print(f"   Chunk overlap  : {CHUNK_OVERLAP} characters")

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3: Generate embeddings and store in ChromaDB
# ─────────────────────────────────────────────────────────────

def build_vector_store(chunks: list, chroma_dir: str) -> Chroma:

    # If a previous vector store exists, delete it so we start fresh.
    # This prevents stale/duplicate chunks if you update your HR docs.
    if os.path.exists(chroma_dir):
        print(f"\n  Deleting the existing vector store found at '{chroma_dir}'. Deleting and rebuilding a fresh store to avoid duplicate chunks.")
        shutil.rmtree(chroma_dir)
        

    print(f"\n Loading embedding model: '{EMBEDDING_MODEL}'")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"\n Embedding {len(chunks)} chunks and saving to '{chroma_dir}'.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )

    print(f"\n Vector store saved to '{chroma_dir}'.")
    print(f"   Total vectors stored: {vector_store._collection.count()}")

    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 4: Sanity check — test a sample retrieval
# ─────────────────────────────────────────────────────────────

def test_retrieval(vector_store: Chroma) -> None:
    
    test_query = "How many days of sick leave are employees entitled to?"

    print(f"\n Test retrieval query: '{test_query}'")
    results = vector_store.similarity_search(test_query, k=2)

    if not results:
        print("  No results returned.")
        return

    print(f"   Returned {len(results)} chunk(s):\n")
    for i, doc in enumerate(results, 1):
        print(f"   ── Chunk {i} ──────────────────────────────")
        print(f"   Source : {doc.metadata.get('source', 'unknown')}")
        print(f"   Page   : {doc.metadata.get('page', 'N/A')}")
        print(f"   Text   : {doc.page_content[:200]}...")
        print()


# ─────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1
    documents = load_documents(DOCS_DIR)

    # Step 2
    chunks = split_documents(documents)

    # Step 3
    vector_store = build_vector_store(chunks, CHROMA_DIR)

    # Step 4
    test_retrieval(vector_store)

    print("\n" + "=" * 55)
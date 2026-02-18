"""
Part 3: RAG System — Warehouse Robot AI
Uses Groq LLM + HuggingFace sentence embeddings + FAISS vector store
to retrieve handling instructions from warehouse documentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load .env from the project root (one level above part3_rag/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate keys at import time so errors are clear
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is missing. Add it to the .env file.")
if not HUGGINGFACEHUB_API_TOKEN:
    raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN is missing. Add it to the .env file.")

# Expose token so HuggingFace libraries can auto-detect it
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# ---------------------------------------------------------------------------
# LangChain imports (new-style packages)
# ---------------------------------------------------------------------------
try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain_community.chains import RetrievalQA
    except ImportError:
        from langchain_classic.chains import RetrievalQA

from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOCS_FOLDER = _PROJECT_ROOT / "docs"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"            # Groq-hosted model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Lazy singleton — initialised on first call to query_rag()
# ---------------------------------------------------------------------------
_qa_chain = None


def _load_documents() -> list[str]:
    """Read every .txt file inside the docs/ folder."""
    documents = []
    if not DOCS_FOLDER.exists():
        raise FileNotFoundError(f"Docs folder not found at {DOCS_FOLDER}")
    for filepath in sorted(DOCS_FOLDER.glob("*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(content)
    if not documents:
        raise ValueError("No .txt documents found in the docs/ folder.")
    return documents


def _initialize():
    """Build the RAG pipeline: embed docs → FAISS → RetrievalQA chain."""
    global _qa_chain

    # Step 1 — Load & chunk documents
    raw_docs = _load_documents()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n",
    )
    chunks = []
    for doc in raw_docs:
        chunks.extend(splitter.split_text(doc))

    # Step 2 — Embed chunks into FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Step 3 — Create Groq-backed LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=512,
    )

    # Step 4 — Build RetrievalQA chain
    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
    )


def query_rag(question: str) -> str:
    """
    Query the RAG pipeline with a natural-language question.
    Initialises the pipeline on first call (lazy loading).

    Args:
        question: The user's question about warehouse operations.

    Returns:
        A string answer derived from the warehouse documentation.
    """
    global _qa_chain
    if _qa_chain is None:
        _initialize()
    
    # Add layman instructions to the question
    layman_question = (
        f"Answer the following question in a simple, clear way for someone who isn't a technical expert. "
        f"Avoid jargon and use bullet points where helpful: {question}"
    )
    
    result = _qa_chain.invoke({"query": layman_question})
    # invoke() returns a dict with 'query' and 'result' keys
    return result.get("result", str(result))


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "How should I handle fragile items in the warehouse?",
        "What are the safety protocols for hazardous materials?",
        "What is the maximum weight the gripper can hold?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = query_rag(q)
        print(f"A: {answer}")
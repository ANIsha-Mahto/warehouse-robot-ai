import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load documents (Paragraph chunking)
# -----------------------------
docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
docs_path = os.path.abspath(docs_path)

documents = []
doc_sources = []

for file in os.listdir(docs_path):
    if file.endswith(".txt"):
        with open(os.path.join(docs_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            paragraphs = text.split("\n")

            for para in paragraphs:
                chunk = para.strip()
                if len(chunk) > 30:
                    documents.append(chunk)
                    doc_sources.append(file)

print("Total chunks created:", len(documents))

# -----------------------------
# Create embeddings + FAISS index
# -----------------------------
embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


# -----------------------------
# RAG Query Function (No Generator)
# -----------------------------
def query_rag(question, k=3):

    question_lower = question.lower()

    keyword_matches = []
    keyword_sources = []

    # 1️⃣ Keyword filtering
    for doc, source in zip(documents, doc_sources):
        if any(word in doc.lower() for word in question_lower.split()):
            keyword_matches.append(doc)
            keyword_sources.append(source)

    if keyword_matches:
        retrieved_chunks = keyword_matches[:k]
        source_file = keyword_sources[0]
    else:
        # 2️⃣ Semantic fallback
        question_embedding = embedding_model.encode([question])
        D, I = index.search(np.array(question_embedding), k)
        retrieved_chunks = [documents[i] for i in I[0]]
        source_file = doc_sources[I[0][0]]

    print("\nRetrieved Chunks:")
    for chunk in retrieved_chunks:
        print("----")
        print(chunk)

    # Direct structured response (no LLM generation)
    answer = "\n".join([f"- {chunk.strip()}" for chunk in retrieved_chunks])

    return answer, source_file


# -----------------------------
# Main Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("Ask a question (or type 'exit'): ")

        if user_query.lower() == "exit":
            break

        answer, source = query_rag(user_query)

        print("\nResponse:\n")
        print(answer)
        print(f"\n(Source: {source})\n")

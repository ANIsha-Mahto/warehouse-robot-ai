import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# --- Step 1: Load documents ---
docs_folder = r"part3_rag/docs"
documents = []
for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(docs_folder, filename), "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)

# --- Step 2: Split documents into chunks ---
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))

# --- Step 3: Embed and store in FAISS ---
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

# --- Step 4: Create retrieval QA chain ---
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever(search_kwargs={"k":3}))

# --- Step 5: Query function ---
def query_rag(question):
    return qa.run(question)

# // fixing the changes
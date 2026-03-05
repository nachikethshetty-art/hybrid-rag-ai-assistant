import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# import your ingestion functions
from app.rag.ingest import load_documents, split_documents

# load env variables
load_dotenv()

# project base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


def create_vector_store():
    print("Loading documents...")
    docs = load_documents()

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": os.getenv("HF_TOKEN")}
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving FAISS index...")
    vectorstore.save_local(VECTOR_PATH)

    print("✅ Vector store created successfully!")


if __name__ == "__main__":
    create_vector_store()
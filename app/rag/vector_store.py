import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import ingestion functions
from app.rag.ingest import load_documents, split_documents


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Base directory path
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


# --------------------------------------------------
# Create FAISS Vector Store
# --------------------------------------------------

def create_vector_store():

    print("📄 Loading documents...")
    docs = load_documents()

    if not docs:
        print("❌ No documents found.")
        return

    print(f"✅ Loaded {len(docs)} documents")


    print("✂️ Splitting documents into chunks...")
    chunks = split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")


    print("🧠 Creating embeddings model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    print("⚡ Building FAISS index...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )


    print("📦 Saving FAISS vector store...")

    os.makedirs(VECTOR_PATH, exist_ok=True)

    vectorstore.save_local(VECTOR_PATH)


    print("🎉 Vector store created successfully!")
    print(f"📂 Saved at: {VECTOR_PATH}")


# --------------------------------------------------
# Run Script
# --------------------------------------------------

if __name__ == "__main__":
    create_vector_store()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


def load_documents():
    documents = []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} folder not found")

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"Loading: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_db(chunks):

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    print("Saving vector database...")

    vector_db.save_local(VECTOR_PATH)

    print("Vector database saved successfully!")


if __name__ == "__main__":

    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    create_vector_db(chunks)
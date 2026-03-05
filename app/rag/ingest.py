import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")


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


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # preview first chunk
    if chunks:
        print("\nSample chunk preview:\n")
        print(chunks[0].page_content[:500])
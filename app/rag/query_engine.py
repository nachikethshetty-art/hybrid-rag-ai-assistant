import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from tavily import TavilyClient


# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ---------------------------------------------------
# Initialize Tavily client
# ---------------------------------------------------

tavily = TavilyClient(api_key=TAVILY_API_KEY)


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


# ---------------------------------------------------
# Load FAISS Vector Store
# ---------------------------------------------------

def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ---------------------------------------------------
# Create Retrieval QA Chain
# ---------------------------------------------------

def create_qa_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


# ---------------------------------------------------
# Web search using Tavily
# ---------------------------------------------------

def search_web(query):

    print("🌐 Searching internet with Tavily...")

    results = tavily.search(
        query=query,
        max_results=3
    )

    context = ""

    for r in results["results"]:
        context += r["content"] + "\n"

    return context


# ---------------------------------------------------
# Initialize QA Chain and LLM
# ---------------------------------------------------

qa_chain = create_qa_chain()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ---------------------------------------------------
# Hybrid Query System
# ---------------------------------------------------

def hybrid_query(query):

    print("📚 Searching vector database...")

    response = qa_chain.invoke({"query": query})

    answer = response["result"]

    # If RAG gives useful answer
    if answer and "I don't know" not in answer:

        print("📚 Using RAG document answer")

        return answer

    # Otherwise use LLM knowledge
    print("⚡ Using LLM knowledge")

    llm_answer = llm.invoke(query)

    if llm_answer.content:

        return llm_answer.content

    # If still no good answer → use internet
    print("🌐 Using Tavily web search")

    web_data = search_web(query)

    final_prompt = f"""
Use the following internet information to answer the question.

Context:
{web_data}

Question:
{query}
"""

    final_answer = llm.invoke(final_prompt)

    return final_answer.content


# ---------------------------------------------------
# CLI Testing
# ---------------------------------------------------

if __name__ == "__main__":

    print("Hybrid RAG system ready! (type 'exit' to quit)\n")

    while True:

        query = input("Question: ")

        if query.lower() == "exit":
            break

        answer = hybrid_query(query)

        print("\nAnswer:\n", answer)
        print("\n" + "-" * 60 + "\n")
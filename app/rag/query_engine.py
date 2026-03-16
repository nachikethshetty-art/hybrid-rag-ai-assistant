import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# Load API Keys (Streamlit Secrets)
# --------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


# --------------------------------------------------
# Base directory paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


# --------------------------------------------------
# LLM Setup (Groq)
# --------------------------------------------------

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load FAISS Vector DB
# --------------------------------------------------

try:
    vector_db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector DB loaded successfully")

except Exception as e:
    print("Vector DB load error:", e)
    vector_db = None


# --------------------------------------------------
# Tavily Web Search
# --------------------------------------------------

search = TavilySearchResults(k=8)


# --------------------------------------------------
# Question Router
# --------------------------------------------------

def route_question(question: str):

    q = question.lower()

    # Battery / research topics → vector DB
    if any(word in q for word in [
        "battery", "lithium", "li-ion",
        "recycling", "cathode", "anode"
    ]):
        return "vector"

    # News / sports / recent events → web search
    if any(word in q for word in [
        "latest", "today", "news",
        "2024", "2025", "ipl",
        "cricket", "match", "winner"
    ]):
        return "web"

    # Default → LLM
    return "llm"


# --------------------------------------------------
# Vector Retrieval
# --------------------------------------------------

def vector_search(question):

    if not vector_db:
        return ""

    docs = vector_db.similarity_search(question, k=3)

    context = ""

    for doc in docs:
        context += doc.page_content[:500] + "\n\n"

    return context


# --------------------------------------------------
# Tavily Web Search
# --------------------------------------------------

def web_search(question):

    web_context = ""

    try:
        results = search.invoke({"query": question})

        for r in results:
            web_context += f"{r['title']}\n{r['content']}\nSource: {r['url']}\n\n"

    except Exception as e:
        print("Web search error:", e)

    return web_context


# --------------------------------------------------
# LLM Answer
# --------------------------------------------------

def llm_answer(question):

    try:
        response = llm.invoke(question)
        return response.content

    except Exception as e:
        print("LLM error:", e)
        return "LLM service unavailable."


# --------------------------------------------------
# Hybrid Query Engine
# --------------------------------------------------

def hybrid_query(question: str):

    print("\nUser Question:", question)

    route = route_question(question)

    print("Selected Route:", route)


    # ---------------------------
    # VECTOR RAG
    # ---------------------------

    if route == "vector":

        context = vector_search(question)

        prompt = f"""
You are an AI research assistant.

Answer the question using ONLY the research document context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)

        return response.content


    # ---------------------------
    # WEB SEARCH
    # ---------------------------

    elif route == "web":

        web_context = web_search(question)

        prompt = f"""
You are an AI assistant.

Answer ONLY using the WEB SEARCH RESULTS below.
Do not rely on prior knowledge.

WEB SEARCH RESULTS:
{web_context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)

        return response.content


    # ---------------------------
    # GENERAL KNOWLEDGE
    # ---------------------------

    else:

        return llm_answer(question)

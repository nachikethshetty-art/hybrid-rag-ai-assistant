import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# Load API Keys from Streamlit Secrets
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
    temperature=0.3,
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
except Exception as e:
    print("Vector DB load error:", e)
    vector_db = None


# --------------------------------------------------
# Tavily Web Search
# --------------------------------------------------

search = TavilySearchResults(k=5)


# --------------------------------------------------
# Question Router
# --------------------------------------------------

def route_question(question: str):

    q = question.lower()

    # research / battery related → vector db
    if any(word in q for word in [
        "battery", "lithium", "li-ion", "recycling",
        "electrode", "cathode", "anode"
    ]):
        return "vector"

    # latest info → web search
    if any(word in q for word in [
        "latest", "today", "news", "2024", "2025",
        "recent", "current"
    ]):
        return "web"

    # default → LLM knowledge
    return "llm"


# --------------------------------------------------
# Vector RAG
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

    web_results = ""

    try:

        results = search.invoke({"query": question})

        for r in results:
            web_results += f"{r['title']}\n{r['content']}\nSource: {r['url']}\n\n"

    except Exception as e:
        print("Web search error:", e)

    return web_results


# --------------------------------------------------
# LLM Response
# --------------------------------------------------

def llm_answer(question):

    try:
        response = llm.invoke(question)
        return response.content
    except Exception as e:
        print("LLM error:", e)
        return "LLM service unavailable."


# --------------------------------------------------
# Hybrid Query Function
# --------------------------------------------------

def hybrid_query(question: str):

    print("\nUser Question:", question)

    route = route_question(question)

    print("Route selected:", route)


    # ----------------------------------
    # Vector DB Route
    # ----------------------------------

    if route == "vector":

        context = vector_search(question)

        prompt = f"""
You are an AI research assistant.

Use the following research document context to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)
        return response.content


    # ----------------------------------
    # Web Search Route
    # ----------------------------------

    elif route == "web":

        web_context = web_search(question)

        prompt = f"""
You are an AI assistant.

Use the web search results below to answer the question.

WEB RESULTS:
{web_context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)
        return response.content


    # ----------------------------------
    # General Knowledge Route
    # ----------------------------------

    else:

        return llm_answer(question)

import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# API Keys (Streamlit secrets)
# --------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


# --------------------------------------------------
# Base directory
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


# --------------------------------------------------
# LLM Setup
# --------------------------------------------------

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY
)


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load FAISS
# --------------------------------------------------

try:
    vector_db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector DB loaded")

except Exception as e:
    print("Vector DB not available:", e)
    vector_db = None


# --------------------------------------------------
# Tavily Search
# --------------------------------------------------

search = TavilySearchResults(k=10)


# --------------------------------------------------
# Router
# --------------------------------------------------

def route_question(question: str):

    q = question.lower()

    # research documents
    if any(word in q for word in [
        "battery", "lithium", "li-ion", "recycling",
        "cathode", "anode"
    ]):
        return "vector"

    # current events / sports / news
    if any(word in q for word in [
        "ipl", "match", "winner", "score",
        "latest", "today", "news", "2024", "2025"
    ]):
        return "web"

    return "llm"


# --------------------------------------------------
# Vector Search
# --------------------------------------------------

def vector_search(question):

    if not vector_db:
        return ""

    docs = vector_db.similarity_search(question, k=4)

    context = ""

    for doc in docs:
        context += doc.page_content[:500] + "\n\n"

    return context


# --------------------------------------------------
# Web Search
# --------------------------------------------------

def web_search(question):

    context = ""

    try:

        results = search.invoke({
            "query": question,
            "search_depth": "advanced"
        })

        for r in results:

            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")

            context += f"{title}\n{content}\nSource: {url}\n\n"

    except Exception as e:
        print("Search error:", e)

    return context


# --------------------------------------------------
# Answer using LLM with grounding
# --------------------------------------------------

def grounded_answer(question, context):

    if len(context.strip()) < 50:
        return "I couldn't find reliable information in the retrieved sources."

    prompt = f"""
You are a fact-checking AI assistant.

You MUST follow these rules:

1. Use ONLY the information from the provided sources.
2. Do NOT use prior knowledge.
3. If the answer is not clearly stated, say:
   "I could not find the answer in the sources."
4. Include the source link in your answer.

SOURCES:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        print("LLM error:", e)

        return "LLM service unavailable."


# --------------------------------------------------
# General LLM Answer
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

    print("\nQuestion:", question)

    route = route_question(question)

    print("Route:", route)


    # ------------------------
    # VECTOR RAG
    # ------------------------

    if route == "vector":

        context = vector_search(question)

        return grounded_answer(question, context)


    # ------------------------
    # WEB SEARCH
    # ------------------------

    elif route == "web":

        context = web_search(question)

        return grounded_answer(question, context)


    # ------------------------
    # GENERAL KNOWLEDGE
    # ------------------------

    else:

        return llm_answer(question)

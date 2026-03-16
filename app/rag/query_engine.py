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
# Base directory
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")


# --------------------------------------------------
# LLM Setup (Groq)
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
# Load FAISS Vector DB
# --------------------------------------------------

try:
    vector_db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector DB loaded")

except Exception as e:
    print("Vector DB error:", e)
    vector_db = None


# --------------------------------------------------
# Tavily Search
# --------------------------------------------------

search = TavilySearchResults(k=10)


# --------------------------------------------------
# Question Router
# --------------------------------------------------

def route_question(question: str):

    q = question.lower()

    # Battery / research → vector DB
    if any(word in q for word in [
        "battery", "lithium", "li-ion",
        "recycling", "cathode", "anode"
    ]):
        return "vector"

    # sports / news / latest info → web search
    if any(word in q for word in [
        "ipl", "cricket", "winner", "match",
        "latest", "today", "news", "2024", "2025"
    ]):
        return "web"

    # default → LLM
    return "llm"


# --------------------------------------------------
# Vector Search
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
# Web Search
# --------------------------------------------------

def web_search(question):

    web_context = ""

    try:

        results = search.invoke({
            "query": question,
            "search_depth": "advanced"
        })

        for r in results:

            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")

            web_context += f"{title}\n{content}\nSource: {url}\n\n"

    except Exception as e:
        print("Web search error:", e)

    return web_context


# --------------------------------------------------
# LLM Answer (General Knowledge)
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

    print("Selected route:", route)


    # ------------------------------------
    # VECTOR RAG
    # ------------------------------------

    if route == "vector":

        context = vector_search(question)

        if len(context.strip()) < 30:
            return "No relevant information found in research documents."

        prompt = f"""
You are a research assistant.

Answer ONLY using the research context below.
If the answer is not in the context, say you do not know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)

        return response.content


    # ------------------------------------
    # WEB SEARCH
    # ------------------------------------

    elif route == "web":

        web_context = web_search(question)

        if len(web_context.strip()) < 50:

            return "I couldn't find reliable information from web search results."

        prompt = f"""
You are an AI assistant.

Answer ONLY using the WEB SEARCH RESULTS.
Do NOT use prior knowledge.
Do NOT guess.

If the answer is not clearly in the results,
say "I could not find the answer in search results."

WEB SEARCH RESULTS:
{web_context}

QUESTION:
{question}

ANSWER (include source if possible):
"""

        response = llm.invoke(prompt)

        return response.content


    # ------------------------------------
    # GENERAL KNOWLEDGE
    # ------------------------------------

    else:

        return llm_answer(question)

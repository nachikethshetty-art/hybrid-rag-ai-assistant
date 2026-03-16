import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# Load API keys from Streamlit Secrets
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
# Load FAISS Vector DB safely
# --------------------------------------------------

try:
    vector_db = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print("FAISS loading error:", e)
    vector_db = None


# --------------------------------------------------
# Tavily Web Search
# --------------------------------------------------

search = TavilySearchResults(k=2)


# --------------------------------------------------
# Hybrid Query Function
# --------------------------------------------------

def hybrid_query(question: str):

    print("\nUser Question:", question)

    # ---------------------------
    # 1️⃣ Vector Search
    # ---------------------------

    vector_context = ""

    if vector_db:
        docs = vector_db.similarity_search(question, k=2)

        for doc in docs:
            vector_context += doc.page_content[:400] + "\n\n"


    # ---------------------------
    # 2️⃣ Web Search
    # ---------------------------

    web_results = ""

    try:
        results = search.invoke({"query": question})

        for r in results:
            web_results += r["content"] + "\n\n"

    except Exception as e:
        print("Web search error:", e)


    # ---------------------------
    # 3️⃣ Build Prompt
    # ---------------------------

    final_prompt = f"""
You are a helpful AI research assistant.

Use the context below to answer the user's question.

Rules:
- Prefer local document context if relevant.
- Use web search results for latest information.
- If neither contains the answer, use general knowledge.
- Be concise and clear.

-----------------------------
LOCAL DOCUMENT CONTEXT:
{vector_context}

WEB SEARCH RESULTS:
{web_results}

-----------------------------
USER QUESTION:
{question}

FINAL ANSWER:
"""


    # ---------------------------
    # 4️⃣ LLM Response
    # ---------------------------

    try:
        response = llm.invoke(final_prompt)
        return response.content

    except Exception as e:
        print("Groq error:", e)
        return "Sorry, the AI service is currently unavailable."

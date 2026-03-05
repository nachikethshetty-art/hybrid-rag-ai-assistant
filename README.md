# Hybrid RAG AI Assistant

A **production-style Hybrid Retrieval-Augmented Generation (RAG) AI Assistant** that combines:

* Local document search using **FAISS Vector Database**
* LLM reasoning using **Groq LLM inference**
* Real-time knowledge retrieval using **Tavily Web Search**
* Interactive UI built with **Streamlit**
* Cloud deployment using **Streamlit Cloud**

This system answers questions using:

* Knowledge stored in uploaded documents
* General LLM knowledge
* Live information from the internet

---

# Project Architecture

User Query
↓
Streamlit UI
↓
Hybrid Query Engine

• FAISS Vector Search (Local Documents)
• Groq LLM (General Knowledge)
• Tavily Web Search (Internet)

↓
Final Answer Returned to User

---

# Features

* Hybrid Retrieval System
* Vector database document search
* Real-time internet search
* LLM reasoning
* Fast inference with Groq
* Streamlit user interface
* Cloud deployment ready
* Production-like architecture

---

# Technologies Used

**Python** – Core programming
**LangChain** – LLM orchestration
**FAISS** – Vector similarity search
**Sentence Transformers** – Embeddings
**Groq API** – High speed LLM inference
**Tavily API** – Web search
**Streamlit** – Web interface
**Streamlit Cloud** – Application deployment

---

# Why RAG?

Large Language Models sometimes produce **hallucinations or outdated information**.

RAG solves this problem by:

1. Retrieving relevant documents
2. Passing them as context to the LLM
3. Generating grounded responses

Benefits:

* More accurate answers
* Context-aware responses
* Ability to use custom data

---

# Why FAISS?

FAISS was selected because it provides:

* Extremely fast similarity search
* Efficient vector storage
* Scales to large datasets
* Widely used in production AI systems

---

# Why Ollama Initially?

The project initially used **Ollama**.

Advantages:

* Runs LLM locally
* No API cost
* Privacy-friendly
* Offline capability

However it has limitations:

* Requires large RAM
* Difficult to deploy on cloud servers
* Heavy CPU/GPU requirements

---

# Why We Migrated to Groq

For deployment we switched to **Groq LLM API**.

Advantages:

* Extremely fast inference
* No GPU required on server
* Works on low-cost cloud instances
* Easy cloud deployment

---

# Hybrid Query Strategy

The system processes queries in the following order:

### 1. Vector Search

Search FAISS vector database for relevant document chunks.

### 2. LLM Knowledge

If document context is insufficient, the system uses LLM reasoning.

### 3. Web Search

If the question requires current or external knowledge, **Tavily API** is used.

This ensures **accurate and up-to-date responses**.

---

# Project Structure

hybrid-rag-ai-assistant

app
│
├── rag
│ ├── ingest.py
│ ├── query_engine.py
│ └── vectorstore.py
│
├── ui
│ └── streamlit_app.py
│
└── utils
└── voice_assistant.py

vectorstore

requirements.txt

README.md

---

# Installation

## Clone the repository

git clone https://github.com/nachikethshetty-art/hybrid-rag-ai-assistant.git

cd hybrid-rag-ai-assistant

---

## Create virtual environment

python -m venv .venv

source .venv/bin/activate

---

## Install dependencies

pip install -r requirements.txt

---

# Environment Variables

Create a **.env** file

GROQ_API_KEY=your_key

TAVILY_API_KEY=your_key

---

# Run the Application

streamlit run app/ui/streamlit_app.py

Open browser

http://localhost:8501

---

# Deployment Journey

During development the system was initially deployed on **AWS EC2** to simulate a production cloud environment.

The goal was to run:

User → EC2 Server → Streamlit UI → Hybrid RAG Engine

However several practical challenges occurred:

* Package compatibility issues
* Runtime dependency conflicts
* Streamlit server configuration problems
* Resource limitations on small EC2 instances

Because of these challenges the deployment strategy was adjusted.

---

# Final Deployment: Streamlit Cloud

The application was ultimately deployed using **Streamlit Cloud**, which provides:

* Direct GitHub integration
* Automatic environment setup
* Reliable hosting for Streamlit apps
* Faster deployment for AI prototypes

---

# Live Application

You can try the deployed application here:

https://hybrid-rag-ai-assistant-lcxr4esza6dnwpj7mkpycy.streamlit.app/

The deployed app allows users to:

* Ask questions about uploaded documents
* Retrieve answers using FAISS vector search
* Combine LLM reasoning with Groq inference
* Retrieve real-time information via Tavily web search

---

# Deployment Architecture (Final)

User
↓
Streamlit Cloud
↓
Hybrid RAG Engine

• FAISS Vector Search
• Groq LLM API
• Tavily Web Search

↓
Answer returned to user

---

# Example Questions

* What is machine learning?
* Explain lithium ion batteries
* Latest AI news
* Explain vector databases

---

# Future Improvements

* LangGraph workflow orchestration
* Agent-based reasoning
* Multi-document ingestion
* Authentication system
* Docker deployment
* Kubernetes scaling

---

# Learning Outcomes

This project demonstrates:

* Retrieval Augmented Generation
* Vector search systems
* LLM integration
* Hybrid information retrieval
* AI application deployment
* Cloud infrastructure

---

# Author

Nachiketh S Shetty

Data Science | Machine Learning | AI Systems

GitHub
https://github.com/nachikethshetty-art

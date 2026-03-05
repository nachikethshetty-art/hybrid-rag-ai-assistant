Hybrid RAG AI Assistant

A production-style Hybrid Retrieval-Augmented Generation (RAG) AI Assistant that combines:

• Local document search using FAISS Vector Database
• LLM reasoning using Groq LLM inference
• Real-time knowledge retrieval using Tavily Web Search
• Interactive UI built with Streamlit
• Cloud deployment using AWS EC2

This system answers questions using:

Knowledge stored in uploaded documents

General LLM knowledge

Live information from the internet

Project Architecture

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

Features

• Hybrid Retrieval System
• Vector database document search
• Real-time internet search
• LLM reasoning
• Fast inference with Groq
• Streamlit user interface
• AWS deployment ready
• Production-like architecture

Technologies Used

Python – Core programming
LangChain – LLM orchestration
FAISS – Vector similarity search
Sentence Transformers – Embeddings
Groq API – High speed LLM inference
Tavily API – Web search
Streamlit – Web interface
AWS EC2 – Cloud deployment

Why RAG?

Large Language Models sometimes produce hallucinations or outdated information.

RAG solves this problem by:

Retrieving relevant documents

Passing them as context to the LLM

Generating grounded responses

Benefits:

• More accurate answers
• Context-aware responses
• Ability to use custom data

Why FAISS?

FAISS was selected because it provides:

• Extremely fast similarity search
• Efficient vector storage
• Scales to large datasets
• Widely used in production AI systems

Why Ollama Initially?

The project initially used Ollama.

Advantages:

• Runs LLM locally
• No API cost
• Privacy-friendly
• Offline capability

However it has limitations:

• Requires large RAM
• Difficult to deploy on cloud servers
• Heavy CPU/GPU requirements

Why We Migrated to Groq

For deployment we switched to Groq LLM API.

Advantages:

• Extremely fast inference
• No GPU required on server
• Works on low-cost cloud instances
• Easy cloud deployment

This allowed the system to run efficiently on AWS EC2 Free Tier instances.

Hybrid Query Strategy

The system processes queries in the following order:

Vector Search

Search FAISS vector database for relevant document chunks.

LLM Knowledge

If document context is insufficient, the system uses LLM reasoning.

Web Search

If the question requires current or external knowledge, Tavily API is used.

This ensures accurate and up-to-date responses.

Project Structure

voice-rag-assistant

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

Installation

Clone the repository

git clone https://github.com/yourusername/hybrid-rag-ai-assistant.git

cd hybrid-rag-ai-assistant

Create virtual environment

python -m venv .venv

source .venv/bin/activate

Install dependencies

pip install -r requirements.txt

Environment Variables

Create a .env file

GROQ_API_KEY=your_key

TAVILY_API_KEY=your_key

Run the Application

streamlit run app/ui/streamlit_app.py

Open browser

http://localhost:8501

AWS Deployment

The system can be deployed on AWS EC2.

Architecture

User
↓
AWS EC2
↓
Streamlit Server
↓
Hybrid RAG Engine

• FAISS Vector Search
• Groq LLM API
• Tavily Web Search

Steps

Launch EC2 instance (Ubuntu)

Open ports

22 → SSH
80 → HTTP
8501 → Streamlit

Connect to server

ssh -i key.pem ubuntu@public-ip

Install dependencies

sudo apt update

sudo apt install python3-pip git -y

Clone repository

git clone repo_url

cd repo

Install libraries

pip install -r requirements.txt

Add environment variables

nano .env

Run application

streamlit run app/ui/streamlit_app.py --server.address 0.0.0.0

Access the app

http://public-ip:8501

Example Questions

What is machine learning
Explain lithium ion batteries
Latest AI news
Explain vector databases

Future Improvements

• LangGraph workflow orchestration
• Agent-based reasoning
• Multi-document ingestion
• Authentication system
• Docker deployment
• Kubernetes scaling

Learning Outcomes

This project demonstrates:

• Retrieval Augmented Generation
• Vector search systems
• LLM integration
• Hybrid information retrieval
• AI application deployment
• Cloud infrastructure

Author

Ajit Kumar

Data Science | Machine Learning | AI Systems

GitHub
https://github.com/nachikethshetty-art

If you want, I can also give you 3 powerful improvements that make this repository look like a FAANG-level AI project:

• Architecture diagram
• System workflow diagram
• GitHub badges + screenshots

These dramatically increase recruiter attention.
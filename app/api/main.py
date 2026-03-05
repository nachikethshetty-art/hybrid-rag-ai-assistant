from fastapi import FastAPI
from app.api.schemas import QueryRequest
from app.rag.query_engine import create_qa_chain

app = FastAPI()

qa_chain = create_qa_chain()

@app.get("/")
def home():
    return {"message": "Voice RAG Assistant API Running"}

@app.post("/ask")
def ask_question(request: QueryRequest):

    response = qa_chain({"query": request.question})

    return {
        "question": request.question,
        "answer": response["result"]
    }
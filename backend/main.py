from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import query_rag

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    answer = query_rag(req.question)
    return {"answer": answer}

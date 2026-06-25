import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from chain import ask

load_dotenv()

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG System is running"}

@app.post("/ask")
def ask_question(body: Question):
    result = ask(body.question)
    return result
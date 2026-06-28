import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel, validator
from dotenv import load_dotenv
from chain import ask

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuAsk RAG API",
    description="Multi-document RAG system with RAGAS evaluation",
    version="1.0.0"
)

class Question(BaseModel):
    question: str

    @validator("question")
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Question too short")
        return v.strip()

@app.get("/")
def home():
    return {"message": "DocuAsk RAG System is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Groq LLaMA 3.3 70B",
        "embeddings": "Gemini text-embedding-001",
        "vector_store": "ChromaDB"
    }

@app.post("/ask")
def ask_question(body: Question):
    logger.info(f"Question received: {body.question}")
    start = time.time()

    result = ask(body.question)

    response_time = round((time.time() - start) * 1000, 2)
    result["response_time_ms"] = response_time

    logger.info(f"Answer generated from: {result['sources']} in {response_time}ms")
    return result
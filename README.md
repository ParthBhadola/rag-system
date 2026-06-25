# DocuAsk — Multi-Document RAG System

Ask questions from any PDF and get accurate answers with source citations.
Built with LangChain, ChromaDB, Gemini Embeddings and Groq LLaMA.

---

## What it does

- Upload one or more PDF documents
- Ask any natural language question
- Get answers grounded in your documents with source citation
- Automatically evaluated for faithfulness using RAGAS (score: 0.98)

---

## System Architecture

PDF Upload

↓

Text Extraction — PyMuPDF

↓

Chunking — LangChain (500 chars, 50 overlap)

↓

Embedding — Gemini text-embedding-004

↓

Vector Storage — ChromaDB (persistent)

↓

Question → Embedding → Similarity Search

↓

Top 3 chunks + Question → Groq LLaMA 3.3 70B

↓

Answer with Source Citation

---

## Tech Stack

| Layer | Tool |
|---|---|
| Document parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Gemini text-embedding-001 |
| Vector database | ChromaDB |
| LLM | Groq LLaMA 3.3 70B |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |

---

## Project Structure
rag-system/

parser.py       — PDF text extraction

chunker.py      — Split text into chunks

embedder.py     — Generate and store embeddings

retriever.py    — Semantic search from ChromaDB

chain.py        — RAG chain combining retrieval and LLM

evaluator.py    — RAGAS evaluation pipeline

main.py         — FastAPI backend

app.py          — Streamlit frontend

data/           — PDF storage (gitignored)

chroma_db/      — Vector database (gitignored)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ParthBhadola/rag-system.git
cd rag-system
```

**2. Install dependencies**
```bash
pip install fastapi uvicorn langchain langchain-groq langchain-community
pip install google-genai chromadb pymupdf streamlit ragas datasets
pip install python-dotenv groq
```

**3. Create .env file**
GEMINI_API_KEY
GROQ_API_KEY


**4. Run the app**
```bash
streamlit run app.py
```

**5. Run the API**
```bash
uvicorn main:app --reload
```

**6. Run evaluation**
```bash
python evaluator.py
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.98 |
| Answer Relevancy | measured via RAGAS |

Faithfulness of 0.98 means the system almost never halluccinates —
every answer is grounded in the actual document content.

---

## Key Technical Decisions

**Why ChromaDB over FAISS?**
ChromaDB persists to disk automatically. FAISS is in-memory only —
all vectors are lost on restart.

**Why chunk size 500 with 50 overlap?**
500 characters is roughly one complete idea. The 50 character overlap
ensures sentences at chunk boundaries are never cut off and lose meaning.

**Why RAGAS evaluation?**
Building a RAG system is easy. Knowing whether it actually works
requires measurement. RAGAS scores faithfulness and answer relevancy
automatically using the LLM as a judge.


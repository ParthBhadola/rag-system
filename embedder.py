import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="rag_docs")

def embed_and_store(chunks):
    for chunk in chunks:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )   
        embedding = result.embeddings[0].values
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    from parser import extract_all_pdfs
    from chunker import chunk_text

    docs = extract_all_pdfs("data/")
    all_chunks = []
    for filename, text in docs.items():
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)

    print(f"Total chunks to embed: {len(all_chunks)}")
    embed_and_store(all_chunks)
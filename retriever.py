import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="rag_docs")

def retrieve(question, top_k=3):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"]
        })
    return chunks

if __name__ == "__main__":
    question = "What is logistic regression?"
    results = retrieve(question)
    print(f"Question: {question}\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} from {chunk['source']} ---")
        print(chunk["text"])
        print()
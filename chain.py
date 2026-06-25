import os
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question):
    chunks = retrieve(question)
    
    context = ""
    sources = set()
    for chunk in chunks:
        context += chunk["text"] + "\n\n"
        sources.add(chunk["source"])
    
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": list(sources)
    }

if __name__ == "__main__":
    print("RAG System ready. Ask anything from your documents.")
    print("Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        result = ask(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Source: {', '.join(result['sources'])}")
        print()
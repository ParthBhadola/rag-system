import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import FakeEmbeddings
from chain import ask
from retriever import retrieve

load_dotenv()

def run_evaluation():
    test_questions = [
        "What is logistic regression?",
        "What is gradient descent?",
        "What is a support vector machine?",
        "What is overfitting?",
        "What is the sigmoid function?"
    ]

    questions = []
    answers = []
    contexts = []

    print("Running RAG pipeline on test questions...\n")

    for q in test_questions:
        print(f"Q: {q}")
        result = ask(q)
        chunks = retrieve(q)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in chunks])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts
    })

    groq_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        n=1
    )

    fake_embeddings = FakeEmbeddings(size=768)
    wrapped_llm = LangchainLLMWrapper(groq_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(fake_embeddings)

    print("\nEvaluating with RAGAS...\n")
    scores = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )

    print("\n===== RAGAS Evaluation Results =====")
    df = scores.to_pandas()
    print(f"Faithfulness:     {df['faithfulness'].mean():.2f}")
    print(f"Answer Relevancy: {df['answer_relevancy'].mean():.2f}")
    print("\nScore guide: 0 = worst, 1 = best")

if __name__ == "__main__":
    run_evaluation()
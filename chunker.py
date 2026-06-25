from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, filename):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": f"{filename}_chunk_{i}",
            "text": chunk,
            "source": filename
        })
    
    return docs

if __name__ == "__main__":
    sample = "This is a test sentence. " * 100
    result = chunk_text(sample, "test.pdf")
    print(f"Total chunks created: {len(result)}")
    print(f"\nFirst chunk:\n{result[0]['text']}")
import fitz
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for i, page in enumerate(doc):
        text += f"\n--- Page {i+1} ---\n"
        text += page.get_text()
    doc.close()
    return text

def extract_all_pdfs(folder):
    docs = {}
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            print(f"Reading: {file}")
            docs[file] = extract_text_from_pdf(os.path.join(folder, file))
            print(f"Got {len(docs[file])} characters")
    return docs

if __name__ == "__main__":
    docs = extract_all_pdfs("data/")
    for name, text in docs.items():
        print(f"\n===== {name} =====")
        print(text[:500])
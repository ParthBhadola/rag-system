import streamlit as st
import os
from dotenv import load_dotenv
from parser import extract_text_from_pdf
from chunker import chunk_text
from embedder import embed_and_store
from chain import ask

load_dotenv()

st.set_page_config(
    page_title="DocuAsk",
    page_icon="📑",
    layout="wide"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        
        .main-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: #f0f0f0;
            margin-bottom: 0;
        }
        
        .sub-title {
            font-size: 0.95rem;
            color: #888;
            margin-top: 0.2rem;
            margin-bottom: 2rem;
        }
        
        .source-tag {
            display: inline-block;
            background: #1e1e2e;
            color: #888;
            font-size: 0.78rem;
            padding: 3px 10px;
            border-radius: 20px;
            margin-top: 6px;
            border: 1px solid #333;
        }

        .stChatMessage {
            background: transparent !important;
            border: none !important;
        }

        .stChatInputContainer {
            border-top: 1px solid #2a2a2a;
            padding-top: 1rem;
        }

        .upload-hint {
            font-size: 0.82rem;
            color: #666;
            margin-top: 0.5rem;
        }

        div[data-testid="stSidebar"] {
            background: #111;
            border-right: 1px solid #222;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Your Documents")
    st.markdown("<p class='upload-hint'>Upload one or more PDFs to get started</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Process", use_container_width=True, type="primary"):
            with st.spinner("Reading documents..."):
                for file in uploaded_files:
                    save_path = os.path.join("data", file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    text = extract_text_from_pdf(save_path)
                    chunks = chunk_text(text, file.name)
                    embed_and_store(chunks)
            st.success(f"{len(uploaded_files)} document(s) ready")

    st.divider()
    st.markdown("<p class='upload-hint'>Built with LangChain · ChromaDB · Groq · Gemini Embeddings</p>", unsafe_allow_html=True)

st.markdown("<p class='main-title'>DocuAsk</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Ask anything from your uploaded documents</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
        <div style='color: #555; font-size: 0.9rem; margin-top: 3rem; text-align: center;'>
            Upload a PDF on the left and ask your first question below
        </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "source" in msg:
            st.markdown(f"<span class='source-tag'>📄 {msg['source']}</span>", unsafe_allow_html=True)

question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner(""):
            result = ask(question)
            st.write(result["answer"])
            source_text = ', '.join(result['sources'])
            st.markdown(f"<span class='source-tag'>📄 {source_text}</span>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "source": source_text
    })
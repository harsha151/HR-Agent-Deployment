import os
import shutil
import sys
# Corrected and necessary imports for LangChain 1.x structure
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

# --- Configuration (Paths are explicitly resolved relative to THIS script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME = "data"
CHROMA_DIR_NAME = "chroma_db"
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_DIR_NAME)
CHROMA_PATH = os.path.join(SCRIPT_DIR, CHROMA_DIR_NAME)

# NEW CONFIG: Use an open-source embedding model (free, no quota issues)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_vector_db():
    print("--- Starting Document Loading and Indexing ---")
    
    # Check for GROQ API Key (Needed for the generation part of the app)
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. The index will be created, but the app will fail to run.")

    # 1. Clean previous index
    if os.path.exists(CHROMA_PATH):
        print(f"Removing old index at: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    # 2. Load Documents
    documents = []
    
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: The data folder at '{DATA_PATH}' is empty. Please add PDF or TXT files.")
        sys.exit(1)

    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        
        # Determine loader based on file extension
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".txt") or filename.endswith(".md"):
            print(f"Loading Text/MD: {filename}")
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
        else:
            print(f"Skipping unknown file type: {filename}")

    if not documents:
        print("No supported documents loaded. Check the 'data' folder.")
        sys.exit(1)

    # 3. Split and Chunk (THIS IS THE MISSING LOGIC)
    print(f"Loaded {len(documents)} document pages/splits.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len
    )
    # The 'chunks' variable is defined here, fixing the NameError
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 4. Embed and Store - Using Hugging Face Model
    print(f"Creating HuggingFace Embeddings ({EMBEDDING_MODEL_NAME})...")
    # This model downloads its files locally, avoiding the API quota issue.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Chroma.from_documents now has the 'chunks' variable available
    print(f"Creating Chroma Vector Store in '{CHROMA_PATH}'...")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print("--- Indexing Complete! The Chroma DB is ready for the Streamlit app. ---")

if __name__ == "__main__":
    create_vector_db()
import streamlit as st
import os
import sys
# Corrected and necessary imports for LangChain 1.x structure
from langchain_groq import ChatGroq             # Groq LLM
from langchain_huggingface import HuggingFaceEmbeddings # HuggingFace Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration (Centralized and Robust Paths) ---

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_NAME = "chroma_db"
CHROMA_PATH = os.path.join(SCRIPT_DIR, CHROMA_DIR_NAME)

# Other Configurations
MODEL_NAME = "llama-3.3-70b-versatile" # Groq Model for fast inference
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Must match index_knowledge.py
K_CHUNKS = 3 

SYSTEM_TEMPLATE = """
You are a highly helpful and accurate HR Knowledge Base Agent. Your primary function is to answer questions 
ONLY based on the context provided below.

Instructions:
1. Do NOT use external knowledge. If the context does not contain the answer, state clearly, "I apologize, but I cannot find the answer in the knowledge base."
2. Answer comprehensively but keep it concise and strictly relevant to the question.
3. Your tone should be professional and polite.

Context:
{context}

Question:
{question}
"""

@st.cache_resource
def load_rag_chain():
    """Loads the vector store and sets up the RAG chain. Uses cache to load once."""
    
    # 1. Check for Vector Store
    if not os.path.exists(CHROMA_PATH):
        st.error(f"Vector store not found at '{CHROMA_PATH}'. Did you run `python index_knowledge.py`?")
        return None
        
    # 2. Check for API Key (Groq is mandatory for this script)
    if not os.getenv("GROQ_API_KEY"):
        st.error("FATAL: GROQ_API_KEY environment variable is not set. Please set it.")
        return None
    
    try:
        # 3. Load Embeddings (Local Model)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 4. Load DB and Retriever
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": K_CHUNKS})
        
        # 5. Initialize Groq Chat Model
        llm = ChatGroq(model=MODEL_NAME) # Reads GROQ_API_KEY automatically
        
        # 6. Create the RAG Chain
        prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error loading the RAG chain: {e}")
        return None

# --- Streamlit UI and Main Loop (THIS SECTION IS NOW CORRECTLY DEFINED) ---
def main():
    st.set_page_config(page_title="HR Knowledge Base Agent", layout="wide")
    st.title("📚 HR Knowledge Base Agent")
    st.markdown(f"**Powered by:** **{MODEL_NAME}** and LangChain RAG pipeline.")
    
    rag_chain = load_rag_chain()
    if rag_chain is None:
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about employee policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Agent is searching the HR knowledge base..."):
            try:
                full_response = rag_chain.invoke(prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred during the query: {e}")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        st.error("FATAL: GROQ_API_KEY environment variable is not set. Please set it before running.")
    else:
        main()
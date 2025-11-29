import os
import logging
import streamlit as st

# LangChain / RAG imports
from langchain_groq import ChatGroq                     # Groq LLM
from langchain_huggingface import HuggingFaceEmbeddings # HF embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Paths & Config ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_NAME = "chroma_db"
CHROMA_PATH = os.path.join(SCRIPT_DIR, CHROMA_DIR_NAME)

MODEL_NAME = "llama-3.3-70b-versatile"        # Groq model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"     # Must match index script
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

# ---------- RAG Loader (cached) ----------

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """Load vector store + LLM and build the RAG chain. Cached for performance."""
    logger.info("load_rag_chain() called")

    # 1. Check Chroma DB presence
    if not os.path.exists(CHROMA_PATH):
        msg = (
            f"Vector store not found at '{CHROMA_PATH}'. "
            "Did you run your indexing script locally (index_knowledge.py / index.py) "
            "and commit the 'chroma_db' folder to GitHub?"
        )
        logger.error(msg)
        # Return None instead of raising so Streamlit app doesn't crash
        return None

    # 2. Check Groq API key
    if not os.getenv("GROQ_API_KEY"):
        msg = "GROQ_API_KEY environment variable is not set."
        logger.error(msg)
        return None

    try:
        # 3. Embeddings
        logger.info("Loading HuggingFaceEmbeddings (%s)...", EMBEDDING_MODEL_NAME)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info("Embeddings loaded.")

        # 4. Chroma retriever
        logger.info("Loading Chroma vector store from %s ...", CHROMA_PATH)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": K_CHUNKS})
        logger.info("Chroma retriever ready.")

        # 5. Groq LLM
        logger.info("Initialising Groq Chat model: %s", MODEL_NAME)
        llm = ChatGroq(model=MODEL_NAME, temperature=0)

        # 6. Prompt + chain
        prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain constructed successfully.")
        return rag_chain

    except Exception as e:
        # Log full stack trace to Render logs
        logger.exception("Error loading the RAG chain: %s", e)
        return None


# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="HR Knowledge Base Agent", layout="wide")
    st.title("📚 HR Knowledge Base Agent")
    st.markdown(
        f"**Powered by:** **{MODEL_NAME}** and LangChain RAG pipeline."
    )

    # Load RAG chain with visible spinner
    with st.spinner("Loading knowledge base (RAG chain)..."):
        rag_chain = load_rag_chain()

    if rag_chain is None:
        st.error(
            "Failed to load the knowledge base.\n\n"
            "- Check that `chroma_db/` exists on the server\n"
            "- Ensure `GROQ_API_KEY` is set in Render Environment\n"
            "- See Render **Logs** for the detailed Python error."
        )
        return
    else:
        st.success("Knowledge base loaded successfully. You can ask HR questions now. ✅")

    # ----- Chat history -----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ----- User input -----
    if prompt := st.chat_input("Ask a question about employee policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Agent is searching the HR knowledge base..."):
            try:
                answer = rag_chain.invoke(prompt)
            except Exception as e:
                logger.exception("Error during RAG invocation: %s", e)
                st.error(f"An error occurred while answering your question: {e}")
                return

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()

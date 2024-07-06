from langchain_community.vectorstores import Chroma
from fastembed import TextEmbedding
from .utils import load_config
import streamlit as st
import logging

config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_embedding_function():
    try:
        # Use a default supported model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        logger.info(f"Using embedding model: {model_name}")
        return TextEmbedding(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def retrieve_context(query, top_k=3):
    embeddings = get_embedding_function()
    if embeddings is None:
        logger.error("Failed to initialize embeddings.")
        return ""

    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # Log the number of documents in the vectorstore
        logger.info(f"Number of documents in vectorstore: {vectorstore._collection.count()}")

        docs = vectorstore.similarity_search(query, k=top_k)
        context = "\n".join([doc.page_content for doc in docs])

        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        logger.info(f"Context: {context[:500]}...")  # Log first 500 characters of context

        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Error retrieving context: {str(e)}")
        return ""

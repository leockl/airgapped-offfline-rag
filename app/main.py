import streamlit as st
import sys
import os
import logging
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Document QnA System")

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_documents, get_existing_documents, clear_vectorstore, get_embedding_function, remove_document
from app.model_handler import ModelHandler
from app.rag import retrieve_context
from app.utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

@st.cache_resource
def get_model_handler():
    return ModelHandler(config)

def load_models():
    with st.spinner("Loading models... This may take a few minutes."):
        model_handler = get_model_handler()

        if not model_handler.available_models:
            st.error("No models are available. Please check your configuration and model files.")
        else:
            alert = st.success(f"Models loaded successfully! Available models: {', '.join(model_handler.available_models)}")
            time.sleep(2) # Wait
            alert.empty() # Clear the alert

def main():
    st.title("Lightweight Offline Document RAG")

    # Initialize session state variables
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'chat_enabled' not in st.session_state:
        st.session_state.chat_enabled = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = True
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if not st.session_state.models_loaded:
        load_models()
        st.session_state.models_loaded = True

    col1, col2 = st.columns([1, 2])

    with col1:
        settings_section()

    with col2:
        chat_interface()

def settings_section():
    st.subheader("Settings")

    uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])

    existing_docs = get_existing_documents()
    if existing_docs:
        st.write("Existing documents:")
        for doc in existing_docs:
            # st.write(f"- {doc}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {doc}")
            with col2:
                if st.button(f"Remove", key=f"remove_{doc}"):
                # if st.button(f"Remove {doc}", key=f"remove_{doc}"):
                    if remove_document(doc):
                        st.success(f"Removed {doc}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to remove {doc}")
    else:
        st.write("No existing documents found.")

    model_handler = get_model_handler()
    if model_handler.available_models:
        model_choice = st.selectbox("Choose a model", model_handler.available_models)
        st.session_state.model_choice = model_choice
    else:
        st.error("No models available. Please check your configuration and model files.")
        return

    st.session_state.debug_mode = st.checkbox("Debug Mode")
    st.session_state.use_rag = st.checkbox("Use RAG", value=True)

    if st.button("Process Documents"):
        if uploaded_files or existing_docs:
            process_and_enable_chat(uploaded_files)
        elif st.session_state.use_rag:
            st.warning("No documents found. Please upload documents to use RAG or disable RAG.")
        else:
            st.success("Chat enabled without RAG.")
            st.session_state.chat_enabled = True

def process_and_enable_chat(uploaded_files):
    with st.spinner("Processing documents..."):
        try:
            log_capture = io.StringIO()
            log_handler = logging.StreamHandler(log_capture)
            logger.addHandler(log_handler)

            num_chunks = process_documents(uploaded_files)

            logger.removeHandler(log_handler)
            log_contents = log_capture.getvalue()

            if num_chunks > 0:
                st.success(f"Processed {num_chunks} chunks from {len(uploaded_files)} documents")
                st.session_state.chat_enabled = True
            else:
                st.info("No new documents to process.")
                # st.warning("No chunks were processed. Please check your documents.")

            with st.expander("Processing Logs"):
                st.text(log_contents)
            st.session_state.chat_enabled = True
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            st.error(f"Error processing documents: {str(e)}")
            st.session_state.chat_enabled = False

def chat_interface():
    st.subheader("Chat Interface")
    # alert=st.success("Chat enabled")
    # time.sleep(2)
    # alert.empty()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.chat_enabled or not st.session_state.use_rag:
        handle_chat_input()
    elif not st.session_state.use_rag:
        st.info("RAG is disabled. You can start chatting without document context.")
    else:
        st.info("Please process documents or disable RAG to start chatting.")

def handle_chat_input():
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if st.session_state.use_rag:
                context = get_rag_context(prompt)
            else:
                context = ""

            prompt_template = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"

            if st.session_state.debug_mode:
                with st.expander("LLM Prompt"):
                    st.code(prompt_template)

            try:
                model_handler = get_model_handler()
                for chunk in model_handler.generate_stream(prompt_template, st.session_state.model_choice):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error(f"Error generating response: {str(e)}")
                full_response = "I apologize, but I encountered an error while generating the response."

        st.session_state.messages.append({"role": "assistant", "content": full_response})

def get_rag_context(prompt):
    try:
        context = retrieve_context(prompt, top_k=config['top_k'])
        if st.session_state.debug_mode:
            logger.info(f"RAG Context: {context}")
            with st.expander("RAG Debug Information"):
                st.write("RAG Context:")
                st.code(context)
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Error retrieving context: {str(e)}")
        return ""

if __name__ == "__main__":
    main()

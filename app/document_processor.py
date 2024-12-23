# This file is part of airgapped-offline-rag.
#
# Airgapped Offline RAG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Airgapped Offline RAG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Airgapped Offline RAG. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Vincent Koc (https://github.com/vincentkoc)

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredTSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import shutil
from .utils import load_config
import streamlit as st
import logging
from pathlib import Path

config = load_config()
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.xls': UnstructuredExcelLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.ppt': UnstructuredPowerPointLoader,
    '.csv': CSVLoader,
    '.tsv': UnstructuredTSVLoader
}

@st.cache_resource
def get_embedding_function():
    try:
        return FastEmbedEmbeddings(
            model_name=config['embedding_model'],
            max_length=512,
            doc_embed_type="passage",
            cache_dir="./models"
        )
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Error loading embedding model: {str(e)}")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(cache_folder="./models")

@st.cache_resource
def get_vectorstore():
    embeddings = get_embedding_function()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def get_document_loader(file_path):
    """Get the appropriate loader based on file extension."""
    file_extension = Path(file_path).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    loader_class = SUPPORTED_EXTENSIONS[file_extension]
    
    if file_extension in ['.xlsx', '.xls', '.tsv']:
        return loader_class(file_path, mode="elements")
    else:
        return loader_class(file_path)

def process_documents(uploaded_files, rebuild=False):
    if rebuild:
        clear_vectorstore()

    documents = []
    for file in uploaded_files:
        try:
            file_path = os.path.join(DOCUMENTS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            # Get appropriate loader based on file type
            loader = get_document_loader(file_path)
            
            # Load documents
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages/elements from {file.name}")
            
            # Add source metadata
            for doc in docs:
                doc.metadata['source'] = file.name
            documents.extend(docs)

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}")
            st.error(f"Error processing file {file.name}: {str(e)}")
            continue

    if not documents:
        logger.warning("No documents were loaded.")
        return 0

    chunk_size = int(config['chunk_size'])
    chunk_overlap = min(int(config['chunk_overlap']), chunk_size - 1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks")

    if not texts:
        logger.warning("No text chunks were created after splitting.")
        return 0

    vectorstore = get_vectorstore()
    try:
        vectorstore.add_documents(texts)
        vectorstore.persist()
        logger.info(f"Added {len(texts)} chunks to the vector store")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        raise

    return len(texts)

def get_existing_documents():
    try:
        vectorstore = get_vectorstore()
        # Retrieve all documents
        results = vectorstore.get()
        # Extract unique source names from the results
        documents = list(set(metadata['source'] for metadata in results['metadatas']))

        # Persist the vectorstore
        vectorstore.persist()

        return documents
    except Exception as e:
        logging.error(f"Error retrieving existing documents: {e}")
        return []

def clear_vectorstore():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        logger.info("Cleared Chroma vectorstore.")
        get_vectorstore.clear()

    for file in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    logger.info("Cleared documents directory.")
    return True

def remove_document(document_name):
    try:
        vectorstore = get_vectorstore()

        # Check if the document file exists
        document_path = os.path.join(DOCUMENTS_DIR, document_name)
        if os.path.exists(document_path):
            # Remove the document file
            os.remove(document_path)
            logging.info(f"Removed file: {document_path}")
        else:
            logging.warning(f"Document file not found: {document_path}")

        # Get the ids of the documents to delete from vectorstore
        results = vectorstore.get(where={"source": document_name})
        if results and results['ids']:
            # Delete the documents from vectorstore
            vectorstore.delete(ids=results['ids'])
            logging.info(f"Removed {len(results['ids'])} embeddings for document: {document_name}")

            # Persist the changes
            vectorstore.persist()

            return True
        else:
            logging.warning(f"No embeddings found in vectorstore for document: {document_name}")
            return False
    except Exception as e:
        logging.error(f"Error removing document {document_name}: {e}")
        return False

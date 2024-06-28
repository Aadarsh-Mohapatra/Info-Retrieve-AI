# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# General imports
import sys
sys.path.append('E:\\Github_Repo\\Info-Retrieve-AI')
from __init__ import cfg
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import fitz  # PyMuPDF
import logging
from datetime import datetime
import os

# Importing sentence transformers and Pyngrok
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import openai  # For GPT-4 interactions

# Custom imports from your scripts
from pdf_pipeline import QASystem as PDFQASystem, PDFReader, Ingestion, SemanticCache
from gemini_script import QASystem as GeminiQASystem, BlogIndexer as GeminiBlogIndexer
from gpt4_script import QASystem as GPT4QASystem, BlogIndexer as GPT4BlogIndexer

# Streamlit and Pyngrok for web app
import streamlit as st

# Import and configure external APIs
import google.generativeai as genai

genai.configure(api_key=cfg.GOOGLE_API_KEY)

print("Import successful")

# Define the data_folder path
data_folder = "E:\\Github_Repo\\Info-Retrieve-AI\\data_source"


def load_models(file_path=None):
    models = {}

    # Load PDF-related models only if a file_path is provided
    if file_path:
        reader = PDFReader()
        cache_service = SemanticCache()
        ingestion = Ingestion(semantic_cache=cache_service, file_path=file_path)
        document_ids = ingestion.ingest_documents(file_path)
        pdf_qa_system = PDFQASystem(
            ingestion_pipeline=ingestion, cache_service=cache_service
        )
        models["pdf"] = pdf_qa_system

    # Load Gemini-related models
    indexer = GeminiBlogIndexer(
        url="https://escalent.co/thought-leadership/blog/?industry=automotive-and-mobility",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        },
    )
    gemini_qa_system = GeminiQASystem("gemini-pro", indexer)
    models["gemini"] = gemini_qa_system

    # Load GPT-4-related models
    index = pinecone.Index(
        name="blog-index",
        api_key=cfg.PINECONE_API_KEY,
        host="https://blog-index-ntt4sfk.svc.aped-4627-b74a.pinecone.io",
    )
    indexer_instance = GPT4BlogIndexer(
        url="https://escalent.co/thought-leadership/blog/?industry=automotive-and-mobility",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        },
    )
    gpt4_qa_system = GPT4QASystem(
        "gpt-4", indexer_instance, openai_key=cfg.OPENAI_API_KEY
    )
    models["gpt4"] = gpt4_qa_system

    return models


def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to disk."""
    try:
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def main():
    st.title("Welcome to Info-Retrieve-AI")
    st.header("Your catalyst for progress!!")
    st.sidebar.title("Upload your data here!")

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # If a file is uploaded, save it and allow the user to select it for model initialization
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.sidebar.success("File uploaded and saved successfully!")

            # Dropdown to select from uploaded files (assuming they're saved in a specific directory)
            file_list = os.listdir(data_folder)
            selected_file = st.sidebar.selectbox(
                "Select a PDF file for processing:", file_list
            )

            # Button to initialize models for the selected file
            if st.sidebar.button("Initialize Models", key="init_models_main"):
                with st.spinner("Loading models... Please wait"):
                    file_path = os.path.join(data_folder, selected_file)
                    models = load_models(file_path)
                    if models:
                        st.session_state["models"] = models
                        st.session_state["models_loaded"] = True
                        st.sidebar.success("Models initialized successfully!")
                    else:
                        st.sidebar.error("Failed to initialize models.")
        else:
            st.sidebar.error("Failed to save file.")
    else:
        st.sidebar.info("Please upload a PDF file to proceed.")

    # Initialize or retrieve session state variables
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    if "models_loaded" not in st.session_state:
        st.session_state["models_loaded"] = False

    # Ensure models are loaded before allowing interactions
    if st.session_state["models_loaded"]:
        # Source selection
        source = st.radio(
            "Select Source", ("PDF", "Web URL"), key="source_select"
        )  # Provide different key
        # Display the LLM based on the source selected
        if source == "PDF":
            st.write("LLM: gemini-pro")
        elif source == "Web URL":
            st.write("LLM: gpt-4, gemini-pro")
            st.write("Selected Model: gemini-pro")

        # User input for the query
        query = st.text_input("Ask me anything!", key="query_input")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit", key="quey_input"):  # Provide different key
                start_time = time.time()
                # Ensure the correct model is retrieved from session state
                models = st.session_state.get("models", {})
                pdf_qa_system = models.get("pdf")
                gemini_qa_system = models.get("gemini")
                gpt4_qa_system = models.get("gpt4")

                if source == "PDF" and pdf_qa_system:
                    response = pdf_qa_system.generate_response(query)
                elif source == "Web URL" and gemini_qa_system:
                    response = gemini_qa_system.answer_query(query)
                else:
                    response = "Model not available or not initialized properly."

                elapsed_time = time.time() - start_time
                st.session_state["conversation"].append(("You", query))
                response_text = f"{response}"
                response_time = (
                    f"<small>Response generated in {elapsed_time:.2f} seconds.</small>"
                )
                combined_response = f"{response_text}<br>{response_time}"
                st.session_state["conversation"].append(("System", combined_response))

        with col2:
            if st.button("Clear Chat", key="clear_chat"):  # Provide different key
                st.session_state["conversation"].clear()

        for index in range(len(st.session_state["conversation"]) - 1, -1, -1):
            speaker, line = st.session_state["conversation"][index]
            if speaker == "You":
                st.markdown(
                    f"<div style='text-align:right; color: blue; border-radius: 10px; padding: 10px; margin: 10px; background-color: #f0f0f5;'>{line}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; color: green; border-radius: 10px; padding: 10px; margin: 10px; background-color: #e0ffe0;'>{line}</div>",
                    unsafe_allow_html=True,
                )

    else:
        st.error(
            "Please initialize the models first by clicking the 'Initialize Models' button."
        )


if __name__ == "__main__":
    main()

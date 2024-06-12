# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# General imports
import config as cfg
import pandas as pd
import numpy as np
import requests
import time
import fitz  # PyMuPDF
import logging
from datetime import datetime

# Importing sentence transformers and Pyngrok
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import openai  # For GPT-4 interactions

# Custom imports from your scripts
from pdf_pipeline import QAChain as PDFQAChain, PDFReader, Ingestion, SemanticCache

print("Import successful")
from gemini_script import QASystem as GeminiQASystem, BlogIndexer as GeminiBlogIndexer

print("Import successful")
from gpt4_script import QASystem as GPT4QASystem, BlogIndexer as GPT4BlogIndexer

print("Import successful")

# Streamlit and Pyngrok for web app
import streamlit as st
from pyngrok import ngrok

# Import and configure external APIs
import google.generativeai as genai

genai.configure(api_key=cfg.GOOGLE_API_KEY)

# Set up ngrok
ngrok.set_auth_token(cfg.NGROK_AUTH_TOKEN)


# Assuming model instantiation is done through a function or directly in the import
def load_models():
    # Load PDF-related models
    reader = PDFReader()
    file_path = r"E:\Github_Repo\Info-Retrieve-AI\NVIDIA__Annual_Report.pdf"
    cache_service = SemanticCache()
    ingestion = Ingestion(semantic_cache=cache_service, file_path=file_path)
    document_ids = ingestion.ingest_documents(file_path)
    pdf_qa_chain = PDFQAChain(ingestion_pipeline=ingestion, cache_service=cache_service)

    # Load Gemini-related models
    indexer = GeminiBlogIndexer(
        url="https://escalent.co/thought-leadership/blog/?industry=automotive-and-mobility",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        },
    )
    gemini_qa_system = GeminiQASystem("gemini-pro", indexer)

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

    return pdf_qa_chain, gemini_qa_system, gpt4_qa_system


def main():
    st.title("Welcome to Info-Retrieve AI")
    st.header("Your catalyst for progress!!")

    # Initialize or retrieve session state variables
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    if "models_loaded" not in st.session_state:
        st.session_state["models_loaded"] = False

    # Load models if not already loaded
    if not st.session_state["models_loaded"]:
        if st.button("Initialize Models"):
            with st.spinner("Loading models... Please wait"):
                st.session_state["models"] = load_models()
            st.session_state["models_loaded"] = True
            st.success("Models initialized successfully!")

    # Ensure models are loaded before allowing interactions
    if st.session_state["models_loaded"]:
        # Source selection
        source = st.radio("Select Source", ("PDF", "Web URL"), key="source_select")
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
            if st.button("Submit"):
                pdf_qa_chain, gemini_qa_system, gpt4_qa_system = st.session_state[
                    "models"
                ]
                start_time = time.time()

                # Generate response with a spinner
                with st.spinner("Generating response..."):
                    if source == "PDF":
                        response = pdf_qa_chain.generate_response(query)
                    elif source == "Web URL":
                        response = gemini_qa_system.answer_query(query)

                elapsed_time = time.time() - start_time
                response_text = f"{response}"
                response_time = (
                    f"<small>Response generated in {elapsed_time:.2f} seconds.</small>"
                )
                combined_response = f"{response_text}<br>{response_time}"
                st.session_state["conversation"].append(("You", query))
                st.session_state["conversation"].append(("System", combined_response))

        with col2:
            if st.button("Clear Chat"):
                st.session_state["conversation"] = []

        # Display the conversation history
        for index in range(len(st.session_state["conversation"]) - 1, -1, -1):
            speaker, line = st.session_state["conversation"][index]
            if speaker == "You":
                # User query, right-aligned
                st.markdown(
                    f"<div style='text-align: right; color: blue; border-radius: 10px; padding: 10px; margin: 10px; background-color: #f0f0f5;'>{line}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # System response, left-aligned
                st.markdown(
                    f"<div style='text-align: left; color: green; border-radius: 10px; padding: 10px; margin: 10px; background-color: #e0ffe0;'>{line}</div>",
                    unsafe_allow_html=True,
                )

    else:
        st.error(
            "Please initialize the models first by clicking the 'Initialize Models' button."
        )


if __name__ == "__main__":
    main()

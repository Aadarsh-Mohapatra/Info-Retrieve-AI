# -*- coding: utf-8 -*-
"""pdf_pipeline.ipynb

# PDF Code:

## Setup and Installations
"""

import os
import fitz
import time
import warnings
import numpy as np
import pandas as pd
from __init__ import cfg
from typing import List
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores.deeplake import DeepLake
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

os.environ["HUGGINGFACE_TOKEN"] = cfg.HF_TOKEN

# Load Deeplake configuration
DEEPLAKE_API_TOKEN = cfg.DEEPLAKE_API_TOKEN

"""## PDFReader Class"""


class PDFReader:
    """Custom PDF Loader to embed metadata with the pdfs."""

    def __init__(self) -> None:
        self.file_name = ""
        self.total_pages = 0
        self.total_chunks = 0  # Counter for total chunks
        self.total_pages_chunked = 0  # Counter for total pages chunked
        self.model = SentenceTransformer("bert-base-nli-mean-tokens")

    def load_pdf(self, file_path, chunk_size=1000, progress_interval=100):
        # Get the filename from file path
        self.file_name = os.path.basename(file_path)

        # Open the PDF file
        pdf_document = fitz.open(file_path)
        self.total_pages = pdf_document.page_count

        chunks = []

        # Iterate through pages
        for page_number in range(self.total_pages):
            # Extract text content from the page
            page = pdf_document.load_page(page_number)
            page_text = page.get_text()

            # Split the text into chunks
            text_chunks = [
                page_text[i : i + chunk_size]
                for i in range(0, len(page_text), chunk_size)
            ]

            # Encode the entire page text to get text_embedding_page
            text_embedding_page = self.model.encode(page_text)

            # Process each chunk
            start_time = time.time()
            for chunk_number, chunk in enumerate(text_chunks, start=1):
                chunk_embedding = self.model.encode(chunk)
                chunks.append(
                    {
                        "text": chunk,
                        "text_embedding_page": text_embedding_page,
                        "chunk_number": chunk_number,
                        "chunk_text": chunk,
                        "text_embedding_chunk": chunk_embedding,
                        "metadata": {
                            "file_name": self.file_name,
                            "page_no": str(page_number + 1),
                            "total_pages": str(self.total_pages),
                        },
                    }
                )

                # Print progress
                if chunk_number % progress_interval == 0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"Processed {chunk_number}/{len(text_chunks)} chunks in page {page_number+1}. "
                        f"Time elapsed: {elapsed_time:.2f} seconds."
                    )
                    start_time = time.time()

            # Increment total pages chunked
            self.total_pages_chunked += 1

        print(f"Total number of chunks: {self.total_chunks}")
        print(f"Total number of pages chunked: {self.total_pages_chunked}")

        return chunks


"""## Semantic Cache"""


# Semantic Cache Class
class SemanticCache:
    def __init__(self) -> None:
        # Initialize the embeddings model and cache vector store
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
        self.cache_vectorstore = DeepLake(
            dataset_path="database/cache_vectorstore",
            token=DEEPLAKE_API_TOKEN,  # Add Deeplake api
            embedding=self.embeddings,
            read_only=False,
            num_workers=4,
            verbose=False,
        )

    def cache_query_response(self, query: str, response: str):
        # Create a Document object using query as the content and its response as metadata
        doc = Document(
            page_content=query,
            metadata={"response": response},
        )

        # Insert the Document object into cache vectorstore
        _ = self.cache_vectorstore.add_documents(documents=[doc])

    def find_similar_query_response(self, query: str, threshold: int):
        try:
            # Find similar query based on the input query
            sim_response = self.cache_vectorstore.similarity_search_with_score(
                query=query, k=5
            )

            # If sim_response is empty, return an empty list
            if not sim_response:
                return []

            # Return the response from the fetched entry if its score is more than threshold
            return [
                {
                    "response": res[0].metadata["response"],
                }
                for res in sim_response
                if res[1] > threshold
            ]
        except Exception as e:
            raise Exception(e)


"""## Ingestion Class"""


# Define the Ingestion class
class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self, semantic_cache: SemanticCache, file_path: str):
        self.reader = PDFReader()
        self.file_path = file_path
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        # Define and initialize the embeddings attribute
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=cfg.GOOGLE_API_KEY,
        )

        self.semantic_cache = semantic_cache

    def ingest_documents(
        self,
        file: str,
    ):
        # Initialize the vector store
        vstore = DeepLake(
            dataset_path="database/text_vectorstore",
            token=DEEPLAKE_API_TOKEN,  # Add Deeplake api
            embedding=self.embeddings,
            overwrite=True,
            num_workers=4,
            verbose=True,
        )

        # Load PDF and process chunks
        chunks = self.reader.load_pdf(
            self.file_path, progress_interval=100
        )  # added progress interval

        # Ingest the chunks
        ids = vstore.add_texts(
            texts=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "chunk_number": chunk["chunk_number"],
                    "chunk_text": chunk["chunk_text"],
                    "text_embedding_page": chunk["text_embedding_page"],
                    "text_embedding_chunk": chunk["text_embedding_chunk"],
                    "file_name": chunk["metadata"]["file_name"],
                    "page_no": chunk["metadata"]["page_no"],
                    "total_pages": chunk["metadata"]["total_pages"],
                }
                for chunk in chunks
            ],
        )

        # Cache responses in the semantic cache
        for chunk in chunks:
            query = chunk["text"]
            response = chunk["chunk_text"]  # Assuming this is the response
            self.semantic_cache.cache_query_response(query, response)

        return ids


"""## QA System"""

class QASystem:
    def __init__(self, ingestion_pipeline, cache_service) -> None:
        # Initialize Google Generative AI Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=cfg.GOOGLE_API_KEY,
            task_type="retrieval_query",
        )

        # Initialize Gemini Chat model
        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=cfg.GOOGLE_API_KEY,
            convert_system_message_to_human=True,
        )

        # Initialize GPT Cache
        self.cache_service = cache_service

        # Set up ingestion pipeline
        self.ingestion_pipeline = ingestion_pipeline

    def ask_question(self, query: str):
        try:
            # Search for similar query response in cache
            cached_response = self.cache_service.find_similar_query_response(
                query=query, threshold=cfg.CACHE_THRESHOLD
            )

            # If similar query response is present, return it
            if len(cached_response) > 0:
                print("Using cache")
                result = cached_response[0]["response"]
            # Else generate response for the query
            else:
                print("Generating response")
                result = self.generate_response(query=query)
        except Exception as e:
            print("Exception raised. Generating response.")
            result = self.generate_response(query=query)

        return result

    def generate_response(self, query: str):
        try:
            # Initialize the vectorstore and retriever object
            vstore = DeepLake(
                dataset_path="database/text_vectorstore",
                token=DEEPLAKE_API_TOKEN,  # Add Deeplake api
                embedding=self.embeddings,
                read_only=True,
                num_workers=4,
                verbose=False,
            )
            retriever = vstore.as_retriever(search_type="similarity")
            retriever.search_kwargs["distance_metric"] = "cos"
            retriever.search_kwargs["fetch_k"] = 20
            retriever.search_kwargs["k"] = 15

            # Write prompt to guide the LLM to generate response
            prompt_template = """
            Provide the response along with the source of the text from which your response is derived.
            Tasked with information retrieval-augmented generation, maintain a non-conversational tone.
            If uncertain, respond with "I don't know" instead of providing speculative answers. Keep responses concise, within five sentences.
            Always conclude in the next line with "Thanks for asking!".
            Use only the following context pieces to formulate your response: {context}.
            Context: {context}
            Question: {question}

            Answer:
            """

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            chain_type_kwargs = {"prompt": PROMPT}

            # Create Retrieval QA chain
            qa = RetrievalQA.from_chain_type(
                llm=self.model,
                retriever=retriever,
                verbose=False,
                chain_type_kwargs=chain_type_kwargs,
            )

            # Run the QA chain and store the response in cache
            result = qa({"query": query})["result"]
            self.cache_service.cache_query_response(query=query, response=result)
            print("Response generated")

            return result
        except Exception as e:
            print("Exception raised. Generating response.")
            result = self.generate_response(query=query)
            return result

# -*- coding: utf-8 -*-
"""Gemini_script.ipynb

# IRS - Gemini

## Setup and Installations
"""

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Import necessary libraries
from __init__ import cfg
import pandas as pd
import numpy as np
import requests
import pinecone
import google.generativeai as genai
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# Configure the Gemini API using the key from your config.py
genai.configure(api_key=cfg.GOOGLE_API_KEY)

"""## Web Scrapper"""


class BlogScraper:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

    def scrape(self):
        response = requests.get(self.url, headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            box = soup.find("div", class_="gridbox gridbox-170-970")
            items = box.find_all("div", class_="card-title headingC sans")

            data = []
            for index, item in enumerate(items, start=1):
                title = item.text.strip()
                link = item.find("a")["href"]
                link_response = requests.get(link, headers=self.headers)
                if link_response.status_code == 200:
                    link_soup = BeautifulSoup(link_response.content, "html.parser")
                    content = (
                        link_soup.find("div", class_="wysiwyg")
                        .get_text(separator="\n")
                        .strip()
                    )
                    data.append(
                        {
                            "Index": index,
                            "Heading": title,
                            "Hyperlink": link,
                            "Content": content,
                        }
                    )
                else:
                    print(f"Failed to fetch content for hyperlink: {link}")

            return data
        else:
            print("Failed to fetch the webpage.")
            return None


"""## BlogIndexer"""


class BlogIndexer:
    def __init__(self, url, headers):
        self.scraper = BlogScraper(url, headers)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_name = "blog-index"
        self.index = pinecone.Index(
            name=self.index_name,
            api_key=cfg.PINECONE_API_KEY,
            host="https://blog-index-ntt4sfk.svc.aped-4627-b74a.pinecone.io",
        )
        self.index.describe_index_stats()

    def index_content(self):
        data = self.scraper.scrape()
        if data:
            upsert_data = []
            for item in data:
                combined_text = f"{item['Heading']}. {item['Content']}"
                embedding = self.model.encode(combined_text, convert_to_tensor=False)
                embedding_list = embedding.tolist()
                # Include content in metadata for retrieval in the QA system
                upsert_data.append(
                    (str(item["Index"]), embedding_list, {"content": item["Content"]})
                )
            self.index.upsert(vectors=upsert_data)
            print("Content indexed successfully.")

    def view_scraped_data(self):
        data = self.scraper.scrape()
        for item in data:
            print(item)

    def test_embeddings(self):
        data = self.scraper.scrape()
        for item in data:
            embedding = self.model.encode(
                f"{item['Heading']}. {item['Content']}", convert_to_tensor=False
            )
            print(
                f"Index: {item['Index']}, Heading: {item['Heading']}, Embedding: {embedding[:5]}..."
            )


"""## QA Chain : Gemini AI"""


class QASystem:
    def __init__(self, model_name, indexer_instance):
        self.model = genai.GenerativeModel(model_name)
        self.indexer = indexer_instance
        self.logs = pd.DataFrame(
            columns=["Query", "Response"]
        )  # Initialize an empty DataFrame

    def query_to_embedding(self, query):
        embedding = self.indexer.model.encode(query, convert_to_tensor=False)
        return embedding.tolist()

    def retrieve_context(self, query_embedding, top_k=3):
        query_results = self.indexer.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        documents = []
        if query_results.get("matches"):
            for match in query_results["matches"]:
                documents.append(match["metadata"]["content"])
        return documents

    def answer_query(self, query):
        print("Generating query embedding...")
        query_embedding = self.query_to_embedding(query)
        print("Retrieving context...")
        contexts = self.retrieve_context(query_embedding)

        if not contexts:
            response_text = "I don't know. Thanks for asking!"
        else:
            augmented_query = " ".join(contexts) + "\n\n" + query
            prompt = f"Here is the information I found on the topic:\n{augmented_query}\n\nCan you provide a detailed answer based on the information above?"
            print("Generating response based on the context...")
            response = self.model.generate_content(prompt)
            try:
                response_text = response.candidates[0].content.parts[0].text
            except AttributeError:
                response_text = "Failed to parse the response correctly."

            print("Response generated.")

        # # Log the query and the response in the DataFrame
        # new_log_entry = {"Query": query, "Response": response_text}
        # self.logs = pd.concat(
        #     [self.logs, pd.DataFrame([new_log_entry])], ignore_index=True
        # )
        return response_text

    # def save_logs_to_csv(self, filename="gemini_query_logs.csv"):
    #     self.logs.to_csv(filename, index=False)
    #     print(f"Logs saved to {filename}.")

    # def print_log(self):
    #     if self.logs.empty:
    #         print("No entries in the log.")
    #     else:
    #         print(self.logs)

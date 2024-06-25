# -*- coding: utf-8 -*-
"""GPT4_script.ipynb

# IRS - GPT 4

## Setup and Installations
"""

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Import necessary libraries
import config as cfg
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import os
import openai
import logging
import sys
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# Setup OpenAI API key from your config
os.environ["OPENAI_API_KEY"] = cfg.OPENAI_API_KEY

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


# BlogIndexer
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


"""## QA Chain : GPT4"""


class QASystem:
    def __init__(self, model_name, indexer_instance, openai_key):
        openai.api_key = cfg.OPENAI_API_KEY
        self.model_name = model_name
        self.index_name = "blog-index"
        self.indexer = indexer_instance
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.gpt_qa_log = []
        try:
            self.gpt_df_log = pd.read_csv("gpt_qa_log.csv")
        except FileNotFoundError:
            self.gpt_df_log = pd.DataFrame(columns=["Index", "Query", "Response"])

    def query_to_embedding(self, query):
        return self.embedding_model.encode(query, convert_to_tensor=False).tolist()

    def retrieve_context(self, query_embedding, top_k=3):
        query_results = self.indexer.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        contexts = (
            [match["metadata"]["content"] for match in query_results["matches"]]
            if query_results["matches"]
            else []
        )
        return contexts

    def generate_response(self, query, contexts):
        if not contexts:
            return "I don't know. Thanks for asking!"

        augmented_query = "\n\n---\n\n".join(contexts) + "\n\n---\n\n" + query
        prompt = f"You are a Question and Answering bot designed to answer questions using the provided context. Do not answer questions that are asked outside the context. Here's the user question:\n\n{augmented_query}"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

        return response.choices[0].message["content"]

    def answer_query(self, query):
        print("Generating query embedding...")
        query_embedding = self.query_to_embedding(query)
        print("Retrieving context...")
        contexts = self.retrieve_context(query_embedding)

        print("Generating response based on the context...")
        response_text = self.generate_response(query, contexts)

        # Logging the query and response
        if (
            not any(log["Query"] == query for log in self.gpt_qa_log)
            and not (self.gpt_df_log["Query"] == query).any()
        ):
            gpt_log_entry = {
                "Index": len(self.gpt_df_log) + 1,
                "Query": query,
                "Response": response_text,
            }
            self.gpt_qa_log.append(gpt_log_entry)  # Append to list

            gpt_log_data = pd.DataFrame(
                [gpt_log_entry]
            )  # Create a new DataFrame for the row
            self.gpt_df_log = pd.concat(
                [self.gpt_df_log, gpt_log_data], ignore_index=True
            )  # Concatenate it to the existing DataFrame
        else:
            print("Duplicate query detected; not logging.")

        self.gpt_df_log.to_csv("gpt_qa_log.csv", index=False)  # Save DataFrame to CSV
        print("Response logged successfully.")
        return response_text

    def print_gpt_qa_log(self):
        if self.gpt_df_log.empty:
            print("No entries in the QA log.")
        else:
            print(self.gpt_df_log)

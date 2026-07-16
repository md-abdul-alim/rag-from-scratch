import bs4
import os
import uuid
import requests
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.load import dumps, loads
import tiktoken
from typing import Literal, Optional, Tuple
from langchain_core.documents import Document
from operator import itemgetter
import langchainhub as hub
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from ragatouille import RAGPretrainedModel
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(
    model="qwen-max",  # You can also use "qwen-max", "qwen-turbo", etc.
    api_key=os.getenv("ALIBABA_API_KEY"),
    base_url=os.getenv("ALIBABA_OPENAI_URL"),
    temperature=0,
    max_tokens=100
)

print("\n\n------------- Part 14: ColBERT (Contextualized Late Interaction over BERT) --------------")
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")

RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)

results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
print("results: ", results)

retriever = RAG.as_langchain_retriever(k=3)
results = retriever.invoke("What animation studio did Miyazaki found?")
print("results: ", results)

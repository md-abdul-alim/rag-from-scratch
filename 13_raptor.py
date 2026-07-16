import bs4
import os
import uuid
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
from dotenv import load_dotenv
load_dotenv()

"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

Y.T=https://www.youtube.com/watch?v=jbGchdTL7d0
Paper=https://arxiv.org/pdf/2401.18059
CodeRef=https://github.com/labdmitriy/llm-rag/blob/master/notebooks/rag-from-scratch/13-raptor.ipynb

Ref Docs:
    - https://chatgpt.com/share/6a588d18-35c8-83ee-bf0f-aaeb09f992dd
    - https://chat.qwen.ai/s/cec5062a-aed5-4016-8d6f-5c8d32e25eac?fev=0.2.73
"""

llm = ChatOpenAI(
    model="qwen-max",  # You can also use "qwen-max", "qwen-turbo", etc.
    api_key=os.getenv("ALIBABA_API_KEY"),
    base_url=os.getenv("ALIBABA_OPENAI_URL"),
    temperature=0,
    max_tokens=100
)

print("\n\n------------- Part 12: Multi Representation Indexing --------------")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)

# Make splits
splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

import bs4
import os
import numpy as np
import tiktoken
import langchainhub as hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="",
)

print("\n\n------------- Part 2: Indexing --------------")

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# Count tokens considering ~4 char / token
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """"Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

num_tokens_from_string = num_tokens_from_string(question, "cl100k_base")
print("num_tokens_from_string: ", num_tokens_from_string)

# Text embedding models
embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)

len(query_result)

# Cosine similarity is recommended (1 indicates identical) for OpenAI embeddings.
# No need of use cosine_similarity, it's just for understand. it's already manage internally 
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)

#### INDEXING ####

# Load blog
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
# we can also use  .from_huggingface_tokenizer
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


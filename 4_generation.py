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
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="qwen-max",  # You can also use "qwen-max", "qwen-turbo", etc.
    api_key=os.getenv("ALIBABA_API_KEY"),
    base_url=os.getenv("ALIBABA_OPENAI_URL"),
    temperature=1.0,
    max_tokens=100
)

print("\n\n------------- Part 4: Generation --------------")
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
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())


retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("What is Task Decomposition?")

# new here -------------------------------------------
# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
print("prompt: ", prompt)

# # Chain
# chain = prompt | llm

# # Run
# chain.invoke({"context":docs,"question":"What is Task Decomposition?"})

prompt_hub_rag = hub.pull("rlm/rag-prompt")

print("prompt_hub_rag: ", prompt_hub_rag)
"""
from langsmith import Client
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
"""
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")


import bs4
import os
from langchain_core.load import dumps, loads
import tiktoken
from operator import itemgetter
import langchainhub as hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="qwen-max",  # You can also use "qwen-max", "qwen-turbo", etc.
    api_key=os.getenv("ALIBABA_API_KEY"),
    base_url=os.getenv("ALIBABA_OPENAI_URL"),
    temperature=0,
    max_tokens=100
)

print("\n\n------------- Part 9: HyDe --------------")
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

# HyDE document generation
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser() 
)

# Run
question = "What is task decomposition for LLM agents?"
generate_docs_for_retrieval.invoke({"question":question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 
retrieved_docs = retrieval_chain.invoke({"question":question})
print("retrieved_docs: ", retrieved_docs)

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retrieved_docs,"question":question})


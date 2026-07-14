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
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

"""
=== MULTI-QUERY RETRIEVAL ===

WHAT IT IS:
Instead of searching the database with just the user's original question, 
we ask the LLM to rewrite it into 3 different versions of the same question.
Then we search the database for ALL 3 versions and combine the unique results.

WHY WE USE IT:
A single question might use words that don't match the words in your documents 
(even if the meaning is the same). By generating multiple perspectives, we 
increase the chance of finding relevant documents that a single query would miss.

EXAMPLE:
User asks: "What is task decomposition for LLM agents?"

The LLM generates 3 alternative questions:
  1. "How do AI agents break down complex tasks?"
  2. "What is the process of dividing tasks for LLMs?"
  3. "Explain task splitting in autonomous agents."

Each question is searched separately in the database. 
Then duplicates are removed, giving us a richer set of documents.

HOW IT WORKS IN THIS CODE:
  generate_queries   -> LLM rewrites the question into 3 versions (split by "\n")
  retriever.map()    -> Searches the database for EACH of the 3 questions
  get_unique_union() -> Flattens results and removes duplicate documents
"""

llm = ChatOpenAI(
    model="qwen-max",  # You can also use "qwen-max", "qwen-turbo", etc.
    api_key=os.getenv("ALIBABA_API_KEY"),
    base_url=os.getenv("ALIBABA_OPENAI_URL"),
    temperature=1.0,
    max_tokens=100
)

print("\n\n------------- Part 5: Multi Query --------------")
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

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives  # 👈 Tells the LLM: "Generate 3 versions of the input question"
    | llm                # 👈 LLM generates the 3 questions as text
    | StrOutputParser()  # 👈 Cleans the LLM output
    | (lambda x: x.split("\n")) # 👈 Splits the text into a Python list: ["Q1", "Q2", "Q3"]
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"

# 2. How & Where is Multi-Query Retrieval working?
# Multi-Query Retrieval technique. It's not a hidden function; it's the explicit pipeline you built.

retrieval_chain = generate_queries | retriever.map() | get_unique_union
#                   ↑                       ↑                  ↑
#              STEP 1: Generate |        STEP 2:     |       STEP 3:
#              3 Questions      | Now Search in DB   |  As from DB get 3
#             by given user's   |    3 timesin       | results from parallal
#             question by LLM   | parallel which     | search so need to check  
#                               | generated step 1   | Deduplicate & Flatten


docs = retrieval_chain.invoke({"question":question})
len(docs)

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})

"""
=== FINAL RAG CHAIN EXPLANATION ===

OVERALL FLOW:
1. Takes input dictionary: {"question": "What is task decomposition?"}
2. Splits into two parallel paths:
   - "context" path: Sends the full dictionary to `retrieval_chain` to fetch documents.
   - "question" path: Uses `itemgetter("question")` to extract just the raw string.
3. Merges the outputs into a new dictionary: {"context": [docs], "question": "string"}
4. Passes this merged dictionary to the `prompt`, then `llm`, then `StrOutputParser`.

WHY itemgetter("question") IS NEEDED:
The `retrieval_chain` expects a dictionary ({"question": "..."}).
But the final `prompt` expects a plain string to fill the {question} variable.
`itemgetter("question")` extracts the raw string from the dictionary so the 
prompt receives the correct data type and doesn't crash.

also more for understand,
If our input is {"question": "What is AI?", "user_id": 123}, calling itemgetter("question") 
will strip away the dictionary and return just the raw string: "What is AI?".
"""
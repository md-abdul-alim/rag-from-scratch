# rag-from-scratch


 - LangChain Expression Language (LCEL)

 # What is the use of OpenAIEmbeddings()?
    - OpenAIEmbeddings Its primary purpose is to convert text into numerical vectors (embeddings).

    - Why is this important?
    Without embeddings, you could only do keyword search (finding exact word matches). With embeddings, you can do semantic search:
    Example:
        User asks: "How do I reset my password?"
        Your document contains: "Steps to recover your account access."
        A keyword search might fail because there are no matching words. 
        But because OpenAIEmbeddings() converted both phrases into vectors, 
        and those vectors are close in semantic space, Chroma will correctly retrieve the relevant document.

# What is the use of vectorstore.as_retriever()?
    - as_retriever() acts as an adapter that converts your specific Vector Database 
    (like Chroma, FAISS, or Pinecone) into a standardized LangChain Retriever object.
    when we want to take our stored documents and feed them into 
    an LLM automatically using LangChain's standardized RAG tools.
    - inside vectorstore.as_retriever(search_kwargs={"k": 1}),
        - search_kwargs tells the retriever exactly how many documents to return when you search.

# What is the use of RunnablePassthrough?
    - A simple component that takes an input and passes it through unchanged to the output.
    The most common place we will see RunnablePassthrough is inside a RunnableParallel (which runs multiple things at the same time and combines their outputs into a dictionary).

    ```
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    # Assume 'retriever' and 'prompt' are already defined

    # 1. Create the chain
    rag_chain = (
        RunnableParallel(
            context=retriever,               # Sends input to retriever -> outputs "context"
            question=RunnablePassthrough()   # Sends input to passthrough -> outputs "question"
        )
        | prompt                             # Combines them into the prompt
        | llm                                # Sends to LLM
    )

    # 2. Invoke the chain
    rag_chain.invoke("What is machine learning?")
    ```
    What happens step-by-step?
        1. Input: The string "What is machine learning?" enters the RunnableParallel.
        2. Branch 1 (context): The string is sent to the retriever. The retriever searches the database and
        outputs a list of Documents. This gets assigned to the key "context".
        3. Branch 2 (question): The string is sent to RunnablePassthrough(). It does absolutely nothing to the string and just outputs "What is machine learning?". This gets assigned to the key "question".
        4. Merge: RunnableParallel combines these into a dictionary:
        {"context": [Document(...)], "question": "What is machine learning?"}
        5. Next Step: This dictionary is passed to the prompt, which successfully formats both the context and the question for the LLM.

        Without RunnablePassthrough(), Branch 2 wouldn't know how to map the raw string input into the "question" key of the dictionary.

    # 3. Advanced Use Case: .assign()
        - RunnablePassthrough has a powerful method called .assign(). It allows you to add new keys to a dictionary without deleting the existing keys.

        ```
        from langchain_core.runnables import RunnablePassthrough

        chain = RunnablePassthrough.assign(
            extra_info=lambda x: f"User asked about: {x['question']}"
        )

        # Input: {"question": "AI", "user_id": 1}
        # Output: {"question": "AI", "user_id": 1, "extra_info": "User asked about: AI"}
        ```

        Purpose: This is incredibly useful for debugging, logging, or injecting static metadata into a pipeline without disrupting the flow of the original data.

        Feature	What it does	When to use it
        RunnablePassthrough()	Returns the exact input unchanged.	Inside RunnableParallel to preserve the original user input (like the question) for the final prompt.
        RunnablePassthrough.assign()	Adds new keys to the input dictionary while keeping the old ones.	When you need to inject extra data, metadata, or debugging info into the middle of a chain.
    
    # 4. Summary Analogy
        Imagine you are running a restaurant kitchen (the LangChain pipeline).
            - You receive an order (the Input Dictionary).
            - You need to give the ingredients to the Chef (the Retriever/LLM).
            - But you also need to keep the original physical receipt (the original input) to give to the cashier.

        RunnablePassthrough is the photocopier. It takes the receipt, makes an exact unchanged copy, and hands it to the cashier, while the original goes to the Chef.

|          Feature      |         What it does          |         When to use it        |
|-----------------------|-------------------------------|-------------------------------|
| RunnablePassthrough() | Returns the exact input unchanged. | Inside RunnableParallel to preserve the original user input (like the question) for the final prompt. |
| RunnablePassthrough.assign() | Adds new keys to the input dictionary while keeping the old ones. | When you need to inject extra data, metadata, or debugging info into the middle of a chain. |
|


# What is the use of StrOutputParser() in rag_chain?
    - It extracts the plain text string from the complex object that the LLM returns.The StrOutputParser acts as a filter at the very end of your chain. It looks at the AIMessage object, grabs the .content attribute, and throws away the rest. Without this we have to manually parse the content. like

    # WITHOUT StrOutputParser
    result = rag_chain.invoke("What is AI?")
    print(result) 
    # Output: AIMessage(content='Artificial Intelligence is...', response_metadata={...})

    print(result.content)


    ```
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```

# What is the use of embd.embed_query()?
    - Convert a single search query (text) into a numerical/mathematical vector (a list of numbers) using OpenAI's embedding model. OpenAI's embedding model (e.g., text-embedding-3-small) analyzes the semantic meaning of the text.
    In a RAG (Retrieval-Augmented Generation) system, we cannot search a Vector Database (like Chroma) using plain English text. The database only understands math. embed_query does this translation.
    We didn't need to call embed_query manually because the retriever does it for you automatically.


    ```
    embd = OpenAIEmbeddings()
    query_result = embd.embed_query(question)
    ```

# embed_query vs. embed_documents

|Method | Purpose |Input |Output |
|-------|---------|------|-------|
|embed_query() | Optimized for a single user search query. |A single string ("What is AI?") |A single list of numbers (1D array). |
| embed_documents() | Optimized for batch-processing multiple documents at once (more efficient API calls). | A list of strings (["Doc 1", "Doc 2"]) | A list of lists of numbers (2D array).|
|

# What is the use of cosine_similarity?
Even though it's not mandatory for a basic RAG pipeline, developers sometimes write this manually for specific reasons:

- Custom Filtering or Reranking
    Sometimes the vector database returns 10 documents, but you want to apply your own strict business logic. For example:

    Only keep documents that are at least 80% similar to the query
    filtered_docs = [
        doc for doc in retrieved_docs 
        if cosine_similarity(query_vector, doc.embedding) > 0.80
    ]

- Debugging
If your RAG chain is returning bad answers, you might manually print the cosine similarity between the query and the retrieved document to see if the retriever is actually finding good matches, or if the problem is with the LLM prompt.

- Building a System Without a Vector Database
If you only have two specific pieces of text and you just want to compare them directly in memory without saving them to a database like Chroma, you would generate both embeddings and run this function.

```
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

query_result = embd.embed_query(question)
document_result = embd.embed_query(document)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
```

# What is top K (The Quantity) most similar documents mean?
The database searches for the 3 most relevant chunks
results = vectorstore.similarity_search("How do I use LCEL?", k=3)

print(results[0].page_content)

here k is simply a number that represents how many results you want the database to return.
If k=1, it returns only the single best match.
If k=3 (the default in many LangChain setups), it returns the 3 best matches.

# Use of direct RecursiveCharacterTextSplitter() vs RecursiveCharacterTextSplitter.from_tiktoken_encoder()? which is better?

|Feature |Standard RecursiveCharacterTextSplitter |.from_tiktoken_encoder |
|--------|----------------------------------------|-----------------------|
|Unit of Measurement |Characters (letters, spaces) |Tokens (AI sub-word units) |
|Speed |Extremely fast (just counts string length). |Slightly slower (must load tokenizer and calculate tokens).|
|Accuracy for LLMs |Low. A rough estimate. 1000 chars ≠= a fixed number of tokens.| High. Exact measurement of what the LLM will actually process. |
|Context Window Safety |Risky. You might accidentally exceed the LLM's token limit. |Safe. You know exactly how many tokens you are feeding the LLM. |
|Best Used For |Quick prototyping, non-LLM text processing. |Production RAG apps, strict context window management. |
|   | The Standard Way: Measuring by Characters | The Precise Way: Measuring by Tokens (via Tiktoken) |
|

We should use .from_tiktoken_encoder() because it splits text based on actual LLM tokens rather than raw characters, ensuring precise alignment with the model's context window limits. This prevents unexpected token overflow errors and allows for accurate calculation of API costs and prompt capacity.

# Use case of chunk_size and chunk_overlap in RecursiveCharacterTextSplitter()

    - chunk_size=300: Limits each piece of text to a maximum of 300 tokens. This guarantees the chunk is small enough to fit safely within the LLM's context window without wasting API costs.
    - chunk_overlap=50: Takes the last 50 tokens of the current chunk and repeats them at the very beginning of the next chunk.
    
Why the overlap matters: If a sentence or key concept is cut exactly in half at the 300-token mark, the overlap ensures the AI still sees the full context in the next chunk, preventing broken or confusing information.

# # embedding follow HNSW graph alogrithm

# get_relevant_documents(old) and invoke (new) both same. both go to database and return result.

# What is multi query retrieval?

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

# What is RAG Fusion?

"""
=== RAG-FUSION EXPLANATION (reciprocal_rank_fusion)===

WHAT IT DOES:
1. Generates 4 different search queries from the original question.
2. Searches the database for all 4 queries (retriever.map()).
3. Merges the 4 result lists using Reciprocal Rank Fusion (RRF).

PURPOSE:
Improves retrieval accuracy by finding documents that consistently rank high 
across multiple query variations, rather than just relying on a single search.

WHY reciprocal_rank_fusion INSTEAD OF get_unique_union?
- get_unique_union ONLY removes duplicates and ignores the order/rank.
- reciprocal_rank_fusion (RRF) calculates a score for each document based on 
  its rank position (Score = 1 / (rank + k)). 
- It keeps documents unique AND sorts them so the most consistently relevant 
  documents (appearing at the top of multiple search results) are prioritized.

WHY RECALCULATE RANK IF INDEXING ALREADY FIXED IT?
- retriever.map() returns 4 SEPARATE lists. A document might be Rank 0 in 
  Query 1, but Rank 2 in Query 2. 
- RRF solves this conflicting ranking by adding up the scores from all 4 lists 
  to create a single "Master Rank" (consensus), identifying the overall best 
  documents across all query perspectives.

ABOUT THE 'k' VALUE (Default = 60):
- Formula: Score = 1 / (rank + k). 
- k=60 is the standard research-backed value. It balances the scoring.
- Lower k (e.g., 10): "Elitist". Top results dominate heavily.
- Higher k (e.g., 100): "Democratic". Lower-ranked docs get more weight.
"""

# To understand a query is "Simple" or ""complex/multi-layered" or how to route query?
- Determining whether a question is "simple" or "complex/multi-layered" is one of the most important challenges in building advanced RAG systems. In the industry, this is called Query Routing or Intent Classification.

You can understand and automate this decision using a combination of linguistic clues (rules of thumb) and programmatic checks (code). Here is how you can do it.

1. Linguistic Clues (How to spot them)
    - Simple Questions (Use Standard RAG)
        - Single Intent: Asks for one specific fact, definition, or piece of information.
        - Direct Lookup: The answer likely exists in a single paragraph or chunk of text.
        - Examples:
            - "What is the refund policy?"
            - "Who is the CEO of the company?"
            - "List the features of the Pro plan."

    - Complex / Multi-layered Questions (Use Decomposition)
        - Multiple Intents: Contains conjunctions like "and", "or", or asks for multiple distinct things.
        - Comparisons: Asks to contrast two or more things.
        - Multi-hop Reasoning: The answer to part A is needed to understand or find part B.
        - Broad Summaries: Asks for a high-level overview of a massive topic.
        - Examples:
            - "Compare the refund policy of the Pro plan with the Basic plan." (Comparison)
            - "What caused the server outage last week, and how much did it cost the company?" (Multi-hop: find cause → find financial impact)
            - "How do I set up the API, and what are the rate limits and error codes I should watch out for?" (Multiple distinct intents)

2. Programmatic Solutions (How to code it)
    We don't want to manually check every query. Instead, you can build a Router at the very beginning of your pipeline to decide which path to take. Here are the 3 best ways to do it:

    - Approach A: The LLM Router (Most Reliable & Recommended)
        - Use a fast, cheap LLM call to classify the question before doing any heavy RAG work.

        ```
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            # 1. Define a routing prompt
            router_prompt = ChatPromptTemplate.from_template("""
            You are an expert at analyzing user questions. 
            Classify the following question as either 'SIMPLE' or 'COMPLEX'.

            - SIMPLE: Asks for a single fact, definition, or direct lookup.
            - COMPLEX: Asks for comparisons, multiple distinct pieces of information, multi-step reasoning, or broad summaries.

            Question: {question}

            Output ONLY the word 'SIMPLE' or 'COMPLEX'.
            """)

            # 2. Create the router chain (use a fast/cheap model like gpt-3.5-turbo or gpt-4o-mini)
            router_chain = router_prompt | llm | StrOutputParser()

            # 3. Route the query
            user_question = "What are the main components of an LLM agent, and how do they compare to traditional software agents?"
            classification = router_chain.invoke({"question": user_question}).strip()

            if classification == "COMPLEX":
                print("Routing to: Decomposition RAG Pipeline")
                # Run the decomposition code you shared earlier
            else:
                print("Routing to: Standard RAG Pipeline")
                # Run standard: retriever.get_relevant_documents(user_question) -> LLM answer
        ```

    - Approach B: The "Sub-question Test" (Self-Assessing)
        - You can ask the LLM to attempt decomposition. If the LLM decides the question can't be broken down (or only generates 1 question), it’s a simple question.

        ```
        # Ask the LLM to generate sub-questions
        sub_questions = generate_queries_decomposition.invoke({"question": user_question})

        # If the LLM only returns 1 question (or the original question), it's simple
        if len(sub_questions) <= 1:
            print("Question is simple. Use standard RAG.")
        else:
            print(f"Question is complex. Breaking into {len(sub_questions)} sub-questions.")
            # Proceed with decomposition loop
        ```

    - Approach C: Keyword / Heuristic Matching (Fastest, but less accurate)
        - If you want to avoid LLM calls entirely for routing, you can use simple Python logic to look for "complexity trigger words".

        ```
            complex_triggers = ["compare", "difference", "versus", "vs", "and how", "and what", "steps to", "entire history", "pros and cons"]

            user_question = user_question.lower()

            # Check if multiple triggers exist or if the question is unusually long
            is_complex = any(trigger in user_question for trigger in complex_triggers) or len(user_question.split()) > 15

            if is_complex:
                print("Routing to: Decomposition RAG")
            else:
                print("Routing to: Standard RAG")
        ```

# What is the Step-Back Feature?
- Step-Back Prompting is a technique where instead of directly answering a specific question, you first generate a more general/abstract version of that question, then use BOTH the original and generalized questions to retrieve context.

- Visual Summary

User Question: "What is task decomposition for LLM agents?"
                    ↓
            ┌───────┴────────┐
            ↓                ↓
    [Specific Query]   [Step-Back Query]
    "task decomposition   "LLM agent design
     for LLM agents"      principles"
            ↓                ↓
    [Retrieve Docs A]  [Retrieve Docs B]
    (specific details)  (foundational concepts)
            ↓                ↓
            └───────┬────────┘
                    ↓
         Combine A + B → Generate Answer

# 🧠 Step-Back RAG: Interview Cheat Sheet

## 1. What is it? (The Elevator Pitch)
Step-Back RAG is an advanced retrieval technique where the system first generates a **broader, more abstract version** of the user's specific query. It then retrieves documents for *both* the specific and the broad query, combining them to provide an answer that is both precise and conceptually grounded.

## 2. How It Works (The 3-Step Pipeline)
1. **Abstract**: An LLM rewrites the specific user query into a general "step-back" question.
2. **Dual Retrieval**: The vector database is queried twice:
   - Query A: The **original** (specific) query.
   - Query B: The **step-back** (general) query.
3. **Synthesis**: The LLM answers the *original* question using the combined context from both retrievals.

## 3. Why Use It? (The Problem It Solves)
Standard RAG suffers from the "missing the forest for the trees" problem. Highly specific queries often miss foundational documents due to keyword mismatches. Step-Back ensures the LLM gets the **specific details** PLUS the **first-principles knowledge** needed to reason correctly.

## 4. When to Use It ✅ vs. Avoid It ❌
- **✅ Use when**: Questions require reasoning, multi-hop logic, or foundational context (e.g., debugging complex errors, medical/legal analysis, explaining "why" something happens).
- **❌ Avoid when**: Simple factual lookup (e.g., "What is the API endpoint for X?"), exact keyword matching is required, or when latency/cost is a strict constraint (it adds 1 extra LLM call + 1 extra retrieval).

---

## 🎯 Interview Example: Standard vs. Step-Back

**User Query**: *"Why is my LLM agent getting stuck in a loop when trying to use the calculator tool?"*

- **Standard RAG Retrieval**: Searches for exact keywords. Might only retrieve a single forum post saying "check your max_iterations parameter." (Misses the bigger picture).
- **Step-Back Query Generated**: *"What are the core principles of tool-use reasoning and loop prevention in LLM agents?"*
- **Step-Back Retrieval**: Finds foundational documentation on the agent's ReAct loop, stop conditions, and tool-schema formatting.
- **Final Result**: The LLM combines the specific error log with the foundational architecture docs to give a root-cause answer: *"Your agent is looping because the calculator tool is returning an unformatted string, which violates the ReAct loop's expected JSON schema, preventing the 'stop' condition from triggering. Fix the tool's output formatter."*

---

## 🗣️ How to Answer in an Interview

**Interviewer**: *"Can you explain Step-Back Prompting in RAG and when you would use it?"*

**Your Answer**: 
"Step-Back RAG is a technique used to improve reasoning in complex queries. Instead of just retrieving documents for a highly specific question, we first prompt an LLM to generate a broader, more abstract 'step-back' version of that question. 

We then perform **dual retrieval**: one for the specific question and one for the general question. Finally, we feed both sets of context to the LLM to answer the original query. 

I would use this when dealing with complex, multi-hop questions where the user needs both specific facts and foundational context to get a correct answer—like debugging a system or explaining a complex domain concept. I would avoid it for simple, direct factual lookups to save on latency and API costs."

# 
- 

# ------------------------------------
# Documents:

- https://towardsdatascience.com/how-to-make-your-llm-more-accurate-with-rag-fine-tuning/
- https://github.com/microsoft/markitdown
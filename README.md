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



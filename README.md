# Types of RAG Architectures:
- These are the clearest and most 14 commonly used types as per technical blogs, documentation:
    - Naive RAG
    - Multimodal RAG
    - Multi-Stage RAG
    - Multi-Agent RAG
    - RAPTOR
    - Corrective RAG
    - Graph RAG
    - Hybrid RAG
    - Adaptive RAG
    - Agentic RAG
    - Advanced RAG
    - Self RAG
    - HYDE
    - ColBERT RAG

---

# RAG Architecture Comparison Matrix 


| Architecture              | Retrieval Steps                          | Typical LLM Calls | Main Strength                                      | Typical Use Case                          | Indexing Cost | Query Cost | Infrastructure Cost | Overall Cost       | Complexity | Latency | Why the Cost?                                             | Best For                       |
| ------------------------- | ---------------------------------------- | ----------------: | -------------------------------------------------- | ----------------------------------------- | ------------- | ---------- | ------------------- | ------------------ | :--------: | :-----: | --------------------------------------------------------- | ------------------------------ |
| **Naive RAG**             | 1                                        |                 1 | Simple and fast retrieval                          | FAQs, small knowledge bases               | Low           | Low        | Low                 | 🟢 **Low**         |      ⭐     |  ⭐⭐⭐⭐⭐  | Single vector search and one LLM call                     | MVPs, prototypes               |
| **Multimodal RAG**        | 1–2                                      |               1–2 | Retrieves text, images, audio, PDFs                | Vision assistants, document understanding | High          | Medium     | High                | 🔴 **High**        |    ⭐⭐⭐⭐    |   ⭐⭐⭐   | Multiple embedding models and multimodal indexing         | Image/PDF/Video QA             |
| **Multi-Stage RAG**       | 2–3                                      |               1–2 | Higher precision through multiple retrieval phases | Large document repositories               | Medium        | High       | Medium              | 🟡 **Medium–High** |     ⭐⭐⭐    |    ⭐⭐   | Multiple retrieval passes and filtering                   | Large-scale search             |
| **Multi-Agent RAG**       | Multiple collaborative retrievals        |             5–20+ | Specialized agents collaborate                     | Enterprise AI workflows                   | High          | Very High  | Very High           | 🔴 **Very High**   |    ⭐⭐⭐⭐⭐   |    ⭐    | Multiple LLM agents, orchestration, memory                | Complex enterprise assistants  |
| **RAPTOR**                | Hierarchical retrieval                   |                 1 | Excellent long-document understanding              | Books, reports, research papers           | Very High     | Medium     | Medium              | 🔴 **High**        |    ⭐⭐⭐⭐    |   ⭐⭐⭐   | Recursive summarization tree built during indexing        | Large document collections     |
| **Corrective RAG (CRAG)** | Retrieval + validation + retry           |               2–3 | Detects and corrects poor retrieval                | High-accuracy QA systems                  | Medium        | High       | Medium              | 🔴 **High**        |    ⭐⭐⭐⭐    |    ⭐⭐   | Retrieval evaluation and fallback retrieval               | Critical applications          |
| **Graph RAG**             | Graph traversal                          |                 1 | Relationship-aware reasoning                       | Legal, healthcare, finance                | Very High     | Medium     | Very High           | 🔴 **High**        |    ⭐⭐⭐⭐    |   ⭐⭐⭐   | Knowledge graph creation and maintenance                  | Connected knowledge domains    |
| **Hybrid RAG**            | Dense + Sparse retrieval                 |                 1 | Higher recall and precision                        | Enterprise search                         | Medium        | Medium     | Medium              | 🟡 **Medium**      |     ⭐⭐⭐    |   ⭐⭐⭐⭐  | Maintains multiple search indexes                         | Production search systems      |
| **Adaptive RAG**          | Dynamic routing (0–2 retrievals)         |               1–2 | Uses retrieval only when needed                    | Mixed-complexity questions                | Medium        | Medium     | Medium              | 🟡 **Medium**      |    ⭐⭐⭐⭐    |   ⭐⭐⭐   | Query complexity detection and routing logic              | Cost-efficient production AI   |
| **Agentic RAG**           | Dynamic multi-step retrieval             |             3–10+ | Planning, reasoning, tool usage                    | AI copilots and autonomous assistants     | Medium        | Very High  | High                | 🔴 **Very High**   |    ⭐⭐⭐⭐⭐   |    ⭐    | Multiple agent decisions, tool calls, iterative retrieval | Autonomous AI systems          |
| **Advanced RAG**          | 1 + reranking                            |               1–2 | Better retrieval quality                           | Production RAG applications               | Low           | Medium     | Low                 | 🟡 **Medium**      |     ⭐⭐     |   ⭐⭐⭐⭐  | Query rewriting, reranking, metadata filtering            | Most production deployments    |
| **Self RAG**              | Iterative retrieval with self-reflection |               3–8 | LLM evaluates its own retrieval                    | Research assistants                       | Medium        | Very High  | Medium              | 🔴 **Very High**   |    ⭐⭐⭐⭐⭐   |    ⭐    | Reflection loops require multiple LLM calls               | Complex reasoning tasks        |
| **HYDE**                  | Hypothetical document → retrieval        |                 2 | Improves retrieval for ambiguous queries           | Low-resource domains, sparse datasets     | Low           | Medium     | Low                 | 🟡 **Medium**      |     ⭐⭐⭐    |   ⭐⭐⭐⭐  | Generates a hypothetical answer before retrieval          | Difficult semantic search      |
| **ColBERT RAG**           | Late-interaction retrieval               |                 1 | Highly accurate semantic retrieval                 | Large-scale search engines                | High          | High       | High                | 🔴 **High**        |    ⭐⭐⭐⭐    |    ⭐⭐   | Token-level embeddings and late interaction scoring       | High-precision semantic search |

---

## Recommendation by Budget

| Budget                                   | Recommended Architectures                    |
| ---------------------------------------- | -------------------------------------------- |
| 💲 **Low Budget**                        | Naive RAG                                    |
| 💲💲 **Moderate Budget**                 | Advanced RAG, Hybrid RAG, Adaptive RAG       |
| 💲💲💲 **Higher Budget**                 | Multi-Stage RAG, Graph RAG, RAPTOR, ColBERT  |
| 💲💲💲💲 **Enterprise / Premium Budget** | CRAG, Self-RAG, Agentic RAG, Multi-Agent RAG |

---

# RAG Concepts

## What is multi query retrieval?

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

## What is RAG Fusion?

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

## Differences Multi-Query RAG vs RAG Fusion.

| Feature             | Multi-Query Retrieval                     | RAG Fusion                                                   |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Purpose             | Generate multiple perspectives of a query | Generate multiple perspectives + intelligently merge results |
| Query Generation    | ✅ Yes                                     | ✅ Yes                                                        |
| Multiple Retrievals | ✅ Yes                                     | ✅ Yes                                                        |
| Ranking Method      | Simple union/deduplication                | Reciprocal Rank Fusion (RRF)                                 |
| Result Quality      | Good                                      | Better                                                       |
| Duplicate Handling  | Remove duplicates                         | Remove duplicates + rank by importance                       |
| Retrieval Cost      | Medium                                    | Medium                                                       |
| Complexity          | Easy                                      | Moderate                                                     |


## What is Decomposition in RAG?

**Decomposition** is an advanced RAG retrieval strategy where a **complex user query is automatically broken down into multiple smaller, simpler sub-questions**. Each sub-question retrieves relevant information independently, and the retrieved results are then combined to generate the final answer.

Instead of asking the vector database one large question, the system asks **multiple focused questions**, which significantly improves retrieval quality.

### ⚙️ How Does the Decomposition Feature Work?
The Decomposition pipeline follows these steps:

### Step 1 — Query Analysis
- The LLM receives the user's original complex query.
- It analyzes the query to identify **distinct information needs**, dependencies, and sub-topics.

### Step 2 — Query Decomposition
- The LLM breaks the original query into **2–N sub-questions**.
- Each sub-question targets a **single, specific piece of information**.
- Sub-questions may be **sequential** (one depends on another) or **parallel** (independent).

**Example:**
- Original Query:
    - "Compare the revenue growth of Company A and Company B in 2024, and explain how their strategies differed."

- Decomposed Sub-Queries:
    - "What was Company A's revenue growth in 2024?"
    - "What was Company B's revenue growth in 2024?"
    - "What was Company A's business strategy in 2024?"
    - "What was Company B's business strategy in 2024?"

### Step 3 — Individual Retrieval
- Each sub-question is **embedded** and sent to the vector store independently.
- Relevant chunks/documents are retrieved for **each** sub-question.
- This yields **more precise and targeted** context than a single broad retrieval.

### Step 4 — (Optional) Iterative / Recursive Decomposition
- If a sub-question is still too complex, it can be **further decomposed** recursively.
- Some implementations use a **tree structure** (e.g., Tree of Thoughts) for deep decomposition.

### Step 5 — Answer Synthesis
- All retrieved contexts from sub-questions are **aggregated**.
- The LLM generates a **unified, coherent final answer** using all gathered evidence.
---

## 🕐 When Should You Use Decomposition?

### ✅ Use Decomposition When:

| Scenario | Why Decomposition Helps |
|---|---|
| **Multi-part questions** | e.g., "What is X, and how does it compare to Y?" — each part needs different documents. |
| **Comparative queries** | Comparing two or more entities requires separate retrieval for each. |
| **Multi-hop reasoning** | Answer requires chaining facts from different sources. |
| **Long / ambiguous queries** | A single embedding can't capture all intents; splitting improves recall. |
| **Domain-specific complex queries** | Legal, medical, or financial questions with multiple clauses. |
| **Low retrieval accuracy** | When single-query retrieval returns irrelevant or partial results. |

### ❌ Avoid Decomposition When:

- The query is **simple and single-intent** (e.g., "What is the capital of France?").
- **Latency is critical** — decomposition adds extra LLM calls and retrieval rounds.
- The knowledge base is **very small** and a single retrieval is sufficient.
- **Cost constraints** are tight (more LLM calls = more tokens = more cost).

---

## 🔑 Key Benefits

- **Higher retrieval precision** — each sub-query matches more relevant chunks.
- **Better coverage** — no part of a complex question is ignored.
- **Reduced hallucination** — grounded answers from targeted evidence.
- **Modular debugging** — you can inspect which sub-query failed.

## ⚠️ Trade-offs

- **Increased latency** — multiple LLM + retrieval calls.
- **Higher token/cost usage** — decomposition + synthesis both consume tokens.
- **Decomposition quality depends on the LLM** — poor splitting leads to poor results.
----

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

-----------

# What is Hypothetical Document Embeddings (HyDE)?
- HyDE stands for Hypothetical Document Embeddings. It's a retrieval technique that improves RAG systems by generating a hypothetical answer to your question first, then using that hypothetical answer to find relevant documents.

Without HyDE:
Question: "What is task decomposition?"
↓ (embed)
[vector: "task", "decomposition"]
↓ (search)
May miss docs that say "breaking down complex objectives"

With HyDE:
Question: "What is task decomposition?"
↓ (LLM generates)
Hypothetical: "Task decomposition breaks complex goals into subtasks..."
↓ (embed)
[vector: "breaks", "complex", "goals", "subtasks"]
↓ (search)
Finds docs with similar answer-language

|Aspect |Traditional RAG |HyDE RAG |
|-------|----------------|---------|
|Speed |Faster (1 LLM call) |Slower (2 LLM calls) |
|Cost |Lower |Higher|
|Recall |Good |Better|
|Complexity |Simple |More complex|
|Best for |Clear queries |Ambiguous/technical queries|
|

# HyDE (Hypothetical Document Embeddings) - Quick Notes

## 1. What is HyDE? (One-Sentence Definition)
HyDE is an advanced RAG technique where an LLM first generates a "fake" (hypothetical) answer to the user's query, and then uses *that generated text* to search the vector database, instead of using the original query.

## 2. The Core Problem It Solves
**The Vocabulary Mismatch:** Users ask questions in casual language, but documents contain answers in formal/technical language. 
- *Query:* "Why is my code crashing?"
- *Document:* "Runtime exceptions occur due to null pointer dereferencing."
Traditional embedding might miss this match. HyDE bridges this gap.

## 3. How It Works (The 4-Step Flow)
1. **Generate:** User asks a question → LLM generates a hypothetical, plausible answer (without looking at the docs).
2. **Embed:** The *hypothetical answer* is converted into a vector embedding.
3. **Retrieve:** This embedding is used to search the vector database for *real* documents that look similar to the hypothetical answer.
4. **Answer:** The retrieved real documents are passed to the LLM to generate the final, factually grounded answer.

## 4. Concrete Example
**User Query:** "How do I make my python script run faster?"

- **Traditional RAG:** Embeds the exact query. Might miss documents that use terms like "optimization", "multiprocessing", or "vectorization".
- **HyDE Approach:** 
  1. LLM generates hypothetical answer: *"To optimize Python performance, you can use multiprocessing, leverage vectorization with NumPy, or profile your code to find bottlenecks."*
  2. System embeds *this* text.
  3. Vector DB successfully retrieves the real document titled "Guide to Python Multiprocessing and NumPy Vectorization".
  4. LLM uses that real doc to give the final, accurate answer.

---

## 5. Interview Q&A Cheat Sheet

**Q: Can you explain what HyDE is and why we use it in RAG?**
**A:** "HyDE stands for Hypothetical Document Embeddings. It’s a retrieval technique designed to solve the 'vocabulary mismatch' problem in RAG. Instead of embedding the user's raw query, we first ask an LLM to generate a hypothetical answer to that query. We then embed this hypothetical answer to search the vector database. Because the hypothetical answer uses similar terminology and structure to the actual source documents, it significantly improves retrieval recall, especially for vague or casually worded queries."

**Q: What are the trade-offs of using HyDE?**
**A:** 
- **Pros:** Higher retrieval recall, better handling of ambiguous queries, bridges the gap between query language and document language.
- **Cons:** Increased latency (adds an extra LLM generation step before retrieval), higher API costs, and a slight risk of the LLM hallucinating a completely wrong direction for the hypothetical answer, which could misguide the retrieval.

**Q: When would you choose NOT to use HyDE?**
**A:** "I would avoid HyDE in low-latency, high-throughput production systems where cost and speed are strict constraints. It's also unnecessary if the user queries are already highly specific and use the exact same domain terminology as the source documents (e.g., exact ID lookups or precise keyword searches)."
------
# What is Logical and Semantic Routing in RAG

## 1. What is Routing in RAG?
In Retrieval-Augmented Generation (RAG), "routing" directs a user's query to the most appropriate data source, index, or retrieval strategy before fetching information. This prevents sending every query to a single massive database, improving accuracy, speed, and cost.

### Logical Routing
Uses deterministic rules, keywords, metadata, or explicit LLM classification to direct a query. It acts like a rule-based switchboard.
- *Mechanism:* Keyword matching, regex, or categorical LLM classification (e.g., "HR", "IT", "Finance").

### Semantic Routing
Uses vector embeddings and similarity search to understand the intent and meaning behind a query, routing it to the most contextually relevant data source.
- *Mechanism:* Query embedding is compared against pre-computed embeddings of different data sources to find the closest semantic match.

---

## 2. When to Use Which?

| Feature | When to Use |
| :--- | :--- |
| **Logical Routing** | • Data is strictly partitioned into distinct domains (e.g., HR vs. IT).<br>• Queries contain clear, predictable keywords.<br>• You need low-latency, low-cost, and deterministic behavior.<br>• Compliance requires strict, auditable routing rules. |
| **Semantic Routing** | • Queries are nuanced, conversational, or ambiguous.<br>• Data domains overlap or are not easily defined by keywords.<br>• You want the system to handle natural language variations without manual rule updates.<br>• You need to dynamically pick the best specialized vector store. |

---

## 3. Pros and Cons

### Logical Routing
- **Pros:** 
  - ⚡ Fast & Cheap (no embedding generation needed for routing).
  - 🔒 Deterministic (easy to predict, test, and debug).
  - 🛠️ Easy to implement with simple rules.
- **Cons:** 
  - 🧱 Brittle (fails on unexpected phrasing, slang, or typos).
  - 📈 High maintenance (requires manual rule updates as topics evolve).

### Semantic Routing
- **Pros:** 
  - 🧠 Flexible & Intelligent (understands intent, synonyms, and context).
  - 📉 Low maintenance (adapts to new phrasing automatically).
  - 🎯 Higher accuracy for complex, multi-domain queries.
- **Cons:** 
  - 💰 Costly & Slower (requires embedding generation and vector search).
  - 🌫️ Non-deterministic (can occasionally misroute; harder to debug).

---

## 4. Real-World Example

**Context:** An Internal Company Assistant with three knowledge bases: `HR_Policies`, `IT_Support`, and `Finance_Expenses`.
**User Query:** *"I paid out of pocket for a client meal, what's the process?"*

### Logical Routing Execution:
- Scans for predefined keywords like "expense", "reimbursement", or "receipt".
- *Result:* If "out of pocket" is not in the keyword list, it might fail or route to a generic fallback. It requires manual rule updates to catch this phrasing.

### Semantic Routing Execution:
- Converts the query into a vector embedding.
- Compares it to the summary embeddings of the three knowledge bases.
- `Finance_Expenses` scores 88% similarity because its documents contain semantically related concepts like "reimbursement", "out of pocket", and "client meals".
- *Result:* Successfully routes to `Finance_Expenses` based on meaning, without needing exact keyword matches.

---

## 5. Best Practice: Hybrid Routing
Production systems often combine both:
1. Use **Logical Routing** first for high-confidence, explicit matches (e.g., specific product IDs or clear department mentions).
2. Fall back to **Semantic Routing** for ambiguous or conversational queries to intelligently infer the best destination.
------
# What is Multi-Representation Indexing (also known as Multi-Vector Retrieval or Parent Document Retriever)?

### What is Multi-Representation Indexing?
- In standard RAG, you split documents into chunks and embed them. This creates a dilemma:
Small chunks have precise embeddings (good for retrieval) but lack surrounding context (bad for LLM generation).
Large chunks have rich context but "diluted" embeddings, making them harder to match with a user's query.
Multi-Representation Indexing solves this by storing two representations of the same document:
A concise representation (like a summary, a hypothetical question, or a small chunk) stored in a Vector Store for highly accurate retrieval.
The original, full parent document stored in a Document/Byte Store to provide complete context to the LLM.

### When Should You Use Multi-Representation RAG Feature?
- You should use Multi-Representation Indexing in the following scenarios:
Long, Dense Documents: When dealing with research papers, legal contracts, or long blog posts (like the Lilian Weng posts in your code) where small chunks lose critical context.
The "Goldilocks" Chunking Problem: When you've tried standard chunking and found that small chunks fail to answer questions fully, but large chunks result in poor retrieval accuracy (low similarity scores).

Mixed Media / Complex Data: This pattern is also heavily used to index a text summary of a table or an image, but retrieve the actual table/image for the LLM to process.
High-Stakes Accuracy: When you need the retrieval step to be as precise as possible (matching against a clean, focused summary) but the generation step to be as informed as possible (reading the full source).

---

### 🎤 Interview Q&A Cheat Sheet

**Interviewer**: "Can you explain Multi-Representation Indexing (or Parent Document Retrieval) and when you would use it?"

**Your Answer**:
"Multi-Representation Indexing solves the classic RAG chunking trade-off: small chunks retrieve well but lack context, while large chunks have context but retrieve poorly. 

It works by maintaining two separate stores linked by a unique ID. During indexing, we generate a concise summary (or small chunk) of a large document. We embed and store this summary in the **Vector Store** for highly accurate search, while storing the full original document in a **Document/Byte Store**. 

During retrieval, the system searches the vector store using the summary. Once it finds the best match, it uses the attached document ID to fetch the *full original parent document* from the byte store, giving the LLM complete context to generate a high-quality answer.

**Example**: 
Imagine a 20-page financial earnings report. If we chunk it by 500 tokens, a query about 'Q3 revenue risks' might match a chunk that only contains a fragment of a sentence, confusing the LLM. With Multi-Representation Indexing, we embed a 2-sentence summary of the entire 'Risk Factors' section. The search easily finds this summary, but the retriever then fetches the *entire* 2-page 'Risk Factors' section to give the LLM full context, resulting in a comprehensive and accurate answer."
-----

# What is RAPTO concept in RAG? V.V.I.
- https://chatgpt.com/share/6a588d18-35c8-83ee-bf0f-aaeb09f992dd
- https://chat.qwen.ai/s/cec5062a-aed5-4016-8d6f-5c8d32e25eac?fev=0.2.73
- https://arxiv.org/pdf/2401.18059
- https://www.youtube.com/watch?v=jbGchdTL7d0
----

# What is ColBERT (Contextualized Late Interaction over BERT)?
- https://arxiv.org/abs/2004.12832
- https://hackernoon.com/lang/bn/how-colbert-helps-developers-overcome-the-limits-of-rag
- https://til.simonwillison.net/llms/colbert-ragatouille
- https://github.com/stanford-futuredata/ColBERT
- https://github.com/AnswerDotAI/RAGatouille


- ColBERT (Contextualized Late Interaction over BERT) is an advanced neural retrieval model designed to make information search both highly accurate and computationally efficient 
www.ultralytics.com. In the context of Retrieval-Augmented Generation (RAG), ColBERT has become a highly important architecture because it elegantly solves the classic trade-off between retrieval speed and accuracy.

### Core Concept: Late Interaction
- Traditional retrieval models in RAG typically fall into two categories:
    - Bi-encoders (e.g., standard dense retrieval like Sentence-BERT): These encode the query and document into single vectors independently and compute a simple cosine similarity. They are very fast but lose granular, token-level details milvus.io.
    - Cross-encoders: These process the query and document together through the model, allowing deep interaction. They are highly accurate but computationally expensive and too slow for large-scale retrieval watercrawl.dev.

ColBERT introduces a third paradigm called Late Interaction medium.com. Instead of compressing an entire document into a single vector, ColBERT encodes both the query and the document into multiple vectors, representing individual tokens 
www.emergentmind.com
. The query and document are encoded independently, and their interaction is delayed until the very end of the process 
www.linkedin.com
.
At query time, ColBERT uses a highly optimized operation called MaxSim (Maximum Similarity) 
developer.ibm.com
. For each token in the query, it finds the most similar token in the document and sums these maximum similarity scores to produce a final, highly accurate relevance score 
apxml.com
.

### Importance of ColBERT in RAG Systems
1. Best of Both Worlds: ColBERT brings together the deep contextual power of cross-encoders and the scalability of bi-encoders. This makes it ideal for RAG pipelines that require both speed and high precision.

2. Superior Retrieval Accuracy: Because it retains token-level representations, it avoids the information bottleneck of single-vector models, leading to significantly better retrieval performance for complex, nuanced, or long documents.

3. Computational Efficiency: Document embeddings can be precomputed and indexed offline. During a RAG query, only the query is encoded, and the MaxSim operation is computationally cheap, enabling scalable, real-time retrieval.

4. Interpretability: The granular, token-level matching of the MaxSim operator allows developers to see exactly which query words matched which document words, providing valuable debugging insights for RAG systems.

5. Hybrid RAG Compatibility: ColBERT inherently captures both lexical exactness and deep semantic meaning, making it a powerful first-stage retriever or re-ranker in hybrid RAG architectures.

### Comparison with Dense Retrieval

| Feature                 | Dense Retrieval       | ColBERT                                |
| ----------------------- | --------------------- | -------------------------------------- |
| Document representation | One vector            | Token vectors                          |
| Query representation    | One vector            | Token vectors                          |
| Similarity              | One cosine similarity | MaxSim over tokens                     |
| Accuracy                | Good                  | Higher, especially for nuanced queries |
| Storage                 | Low                   | Higher                                 |
| Retrieval speed         | Faster                | Slower than dense, but optimized       |
| Memory usage            | Low                   | Higher                                 |
-----

# -----------------------------------------------------------------------------------------


#### LangChain Expression Language (LCEL)

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
----

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
---
# What is the use of embd.embed_query()?
    - Convert a single search query (text) into a numerical/mathematical vector (a list of numbers) using OpenAI's embedding model. OpenAI's embedding model (e.g., text-embedding-3-small) analyzes the semantic meaning of the text.
    In a RAG (Retrieval-Augmented Generation) system, we cannot search a Vector Database (like Chroma) using plain English text. The database only understands math. embed_query does this translation.
    We didn't need to call embed_query manually because the retriever does it for you automatically.


    ```
    embd = OpenAIEmbeddings()
    query_result = embd.embed_query(question)
    ```
---
# embed_query vs. embed_documents

|Method | Purpose |Input |Output |
|-------|---------|------|-------|
|embed_query() | Optimized for a single user search query. |A single string ("What is AI?") |A single list of numbers (1D array). |
| embed_documents() | Optimized for batch-processing multiple documents at once (more efficient API calls). | A list of strings (["Doc 1", "Doc 2"]) | A list of lists of numbers (2D array).|
|
---
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
---
# What is top K (The Quantity) most similar documents mean?
The database searches for the 3 most relevant chunks
results = vectorstore.similarity_search("How do I use LCEL?", k=3)

print(results[0].page_content)

here k is simply a number that represents how many results you want the database to return.
If k=1, it returns only the single best match.
If k=3 (the default in many LangChain setups), it returns the 3 best matches.
---
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


# What is RunnableLambda?
- A wrapper that converts a standard Python function into a LangChain `Runnable` object, allowing it to be used inside an LCEL chain.

- When to use it:
    - Custom Logic: When you need to do something LangChain doesn't have a built-in tool for (e.g., custom math, calling an external non-LLM API, complex dictionary manipulation, or routing logic).
    - Bridging Code: LangChain chains expect Runnable objects. If you have a normal Python function, wrapping it in RunnableLambda makes it "chain-compatible."

---
# What is Re-Indexing? how Re-Indexing work? When should we use Re-Indexing?

## 1. What is Re-Indexing in RAG?

**Re-Indexing** is the process of updating, modifying, or completely rebuilding the vector index used by your RAG application. 

Because vector embeddings are highly sensitive to both the underlying text and the mathematical model used to create them, you cannot simply "edit" an existing vector in place. If the source text changes, or if you change the embedding model, the old vector becomes invalid and must be replaced. Re-indexing is the pipeline that handles this replacement to ensure your RAG system is always retrieving accurate, up-to-date, and optimally formatted information.

---

## 2. How Does Re-Indexing Work?

Re-indexing generally falls into two categories: **Incremental (Delta) Re-indexing** and **Full Re-indexing**. 

### A. The Core Pipeline
Regardless of the method, the re-indexing pipeline usually follows these steps:
1. **Detection:** The system identifies what needs to be updated (e.g., a file was modified, a new document was added, or a global model change was triggered).
2. **Processing:** The text is re-chunked (split into smaller pieces) according to the current strategy.
3. **Embedding:** The chunks are passed through the embedding model to generate new vector representations.
4. **Upserting/Deletion:** The new vectors and metadata are sent to the Vector Database (e.g., Pinecone, Milvus, Qdrant, Weaviate). The system either *upserts* (updates/inserts) the new vectors or *deletes* the obsolete ones.

### B. Incremental Re-Indexing (Delta Updates)
This is used for day-to-day data changes. 
* **How it works:** You track changes in your source data using file hashes (e.g., MD5/SHA), timestamps, or Change Data Capture (CDC). 
* If a document's hash changes, the system deletes the old vectors associated with that document's ID and generates/inserts the new vectors. 
* If a document is deleted in the source, its corresponding vectors are purged from the vector DB.

### C. Full Re-Indexing (Rebuilding)
This is used when structural or foundational changes are made to the RAG pipeline.
* **How it works:** The entire vector database is wiped and rebuilt from scratch using the source documents. 
* **Zero-Downtime (Blue/Green Deployment):** In production, you don't want to take the system offline while rebuilding. You create a *new* index (Index B) in the background while the *old* index (Index A) continues to serve user queries. Once Index B is fully populated and verified, you update your application's routing to point to Index B, and then delete Index A.

---

## 3. When Should We Use Re-Indexing?

You should trigger a re-indexing process in the following scenarios:

### Scenario A: Source Data Changes (Use Incremental Re-Indexing)
* **New Information:** Adding new company policies, updated product manuals, or newly published research papers.
* **Updated Information:** An HR policy changes, or a software documentation page is updated to reflect a new feature.
* **Obsolete Information:** Removing outdated data (e.g., last year's pricing structures or deprecated API documentation) so the LLM doesn't hallucinate based on old facts.

### Scenario B: Embedding Model Upgrades (Use Full Re-Indexing)
* **Why:** Vector spaces are **not compatible** across different models. An embedding generated by an older model exists in a completely different mathematical space than one generated by a newer model. 
* **Action:** If you upgrade your embedding model to get better semantic understanding, you *must* full re-index your entire database.

### Scenario C: Chunking Strategy Optimization (Use Full Re-Indexing)
* **Why:** The way you split documents drastically affects retrieval quality. 
* **Action:** If you realize your current chunk size (e.g., 512 tokens) is too small and losing context, or you want to switch from fixed-size chunking to **Semantic Chunking** or **Parent-Child (Small-to-Big) Chunking**, you must re-chunk and re-embed the entire dataset.

### Scenario D: Metadata and Schema Changes (Use Full or Incremental)
* **Why:** RAG systems often rely on metadata filtering (e.g., `filter: {"department": "HR"}`). 
* **Action:** If you realize you need to add a new metadata tag to all documents (like adding a "document_version" or "author" tag to enable hybrid search filtering), you will need to re-process the documents to extract and attach this new metadata to the vectors.

### Scenario E: Data Cleanup and Deduplication
* **Why:** Over time, vector databases can accumulate "garbage" (duplicate vectors, poorly OCR'd PDFs, empty chunks).
* **Action:** Running a full re-index allows you to apply better data-cleaning pipelines, remove duplicates, and ensure the vector DB only contains high-quality, dense information.

---

## 4. Summary Checklist for RAG Engineers

| Trigger | Re-Indexing Type | Downtime Risk | Cost/Compute |
| :--- | :--- | :--- | :--- |
| **New/Updated Documents** | Incremental | None | Low |
| **Deleted Documents** | Incremental | None | Low |
| **New Embedding Model** | Full (Blue/Green) | None (if Blue/Green) | High |
| **New Chunking Strategy** | Full (Blue/Green) | None (if Blue/Green) | High |
| **Adding Metadata Filters** | Full or Incremental | Low to None | Medium |
-----------------------------------
# Documents:

- https://towardsdatascience.com/how-to-make-your-llm-more-accurate-with-rag-fine-tuning/
- https://github.com/microsoft/markitdown

### Learning Path

A practical progression for mastering retrieval in RAG is:

Dense Retrieval (OpenAI Embeddings, BGE, E5)
Vector Databases (Chroma, FAISS, Pinecone, Weaviate)
Hybrid Search (Dense + BM25)
Reranking (Cross-Encoders, Cohere Rerank, BGE Reranker)
ColBERT / Late Interaction Retrieval
Multi-vector Retrieval (e.g., ColBERTv2, multi-vector document representations)
Agentic and Graph-based Retrieval


Reference:
- https://github.com/langchain-ai/rag-from-scratch/

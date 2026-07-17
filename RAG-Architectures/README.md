

Types:
- There are 8 types of architectures. https://www.facebook.com/share/p/1Bp4e1XDeS/
    1. Naive RAG / Naive
    2. Multimodal RAG
    3. HYDE
    4. Corrective RAG / Corrective RAG
    5. Graph RAG / Graph RAG
    6. Hybrid RAG / Hybrid RAG
    7. Adaptive RAG / Adaptive RAG
    8. Agentic RAG / Agentic RAG

- Others
    - Advanced RAG
    - Multi-Stage RAG
    - Self RAG
    - RAPTOR
    - ColBERT RAG
    - Multi-Agent RAG

# Core 

| Architecture          |               Retrieval Steps | Main Strength                   | Typical Use Case               |
| --------------------- | ----------------------------: | ------------------------------- | ------------------------------ |
| Naive RAG             |                             1 | Simple, fast                    | Small knowledge bases          |
| Advanced RAG          |                 1 + reranking | Better retrieval quality        | Production search              |
| Hybrid RAG            |           Multiple retrievers | High recall and precision       | Enterprise search              |
| Graph RAG             |               Graph traversal | Relationship-aware reasoning    | Healthcare, legal, finance     |
| Multi-Stage RAG       |     Multiple retrieval stages | Improved filtering              | Large document collections     |
| Corrective RAG (CRAG) |        Retrieval + validation | Detects poor retrieval          | High-reliability QA            |
| Self-RAG              |     Iterative self-reflection | Adaptive retrieval decisions    | Research and complex reasoning |
| Adaptive RAG          |               Dynamic routing | Balances speed and quality      | Mixed workloads                |
| Agentic RAG           |                  Agent-driven | Tool use and planning           | Autonomous AI assistants       |
| RAPTOR                |        Hierarchical retrieval | Long-document understanding     | Books, reports, manuals        |
| ColBERT RAG           |    Late-interaction retrieval | Highly accurate semantic search | Large-scale search systems     |
| Multi-Agent RAG       | Multiple collaborating agents | Specialized reasoning           | Complex enterprise workflows   |
|
--------------------------------

| Architecture              | Complexity | Retrieval Quality | Latency |                  Relative Cost                 | Why the Cost?                                          | Best For                    |
| ------------------------- | ---------: | ----------------: | ------: | :--------------------------------------------: | ------------------------------------------------------ | --------------------------- |
| **Naive RAG**             |          ⭐ |               ⭐⭐⭐ |   ⭐⭐⭐⭐⭐ |                   🟢 **Low**                   | One vector search + one LLM call                       | MVPs, FAQs                  |
| **Advanced RAG**          |         ⭐⭐ |              ⭐⭐⭐⭐ |    ⭐⭐⭐⭐ |                  🟡 **Medium**                 | Query rewriting, reranking, metadata filtering         | Production applications     |
| **Hybrid RAG**            |        ⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |     ⭐⭐⭐ |                  🟡 **Medium**                 | Maintains both BM25 and vector indexes                 | Enterprise search           |
| **Graph RAG**             |       ⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |     ⭐⭐⭐ |                   🔴 **High**                  | Knowledge graph creation, maintenance, graph traversal | Healthcare, Legal, Finance  |
| **Multi-Stage RAG**       |        ⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |      ⭐⭐ |               🟡 **Medium–High**               | Multiple retrieval passes and filtering                | Large document collections  |
| **Corrective RAG (CRAG)** |       ⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |      ⭐⭐ |                   🔴 **High**                  | Retrieval evaluation plus fallback retrievals          | High-accuracy systems       |
| **Self-RAG**              |      ⭐⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |       ⭐ |                🔴 **Very High**                | Multiple LLM reasoning and retrieval loops             | Research assistants         |
| **Adaptive RAG**          |       ⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |     ⭐⭐⭐ |                  🟡 **Medium**                 | Smart routing avoids unnecessary retrieval             | Mixed workloads             |
| **Agentic RAG**           |      ⭐⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |       ⭐ |                🔴 **Very High**                | Multiple agent steps, tools, memory, planning          | AI agents                   |
| **RAPTOR**                |       ⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |     ⭐⭐⭐ | 🔴 **High (Indexing)** / 🟡 **Medium (Query)** | Expensive preprocessing, efficient retrieval afterward | Large document repositories |
| **ColBERT RAG**           |       ⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |      ⭐⭐ |                   🔴 **High**                  | Token-level embeddings and late interaction            | High-end semantic search    |
| **Multi-Agent RAG**       |      ⭐⭐⭐⭐⭐ |             ⭐⭐⭐⭐⭐ |       ⭐ |              🔴 **Extremely High**             | Several LLM agents collaborate per request             | Complex enterprise AI       |
|

------------------------------------
## Cost Breakdown

| Architecture    | Indexing Cost |     Query Cost | Infrastructure Cost |      Overall      |
| --------------- | ------------: | -------------: | ------------------: | :---------------: |
| Naive RAG       |           Low |            Low |                 Low |       🟢 Low      |
| Advanced RAG    |           Low |         Medium |                 Low |     🟡 Medium     |
| Hybrid RAG      |        Medium |         Medium |              Medium |     🟡 Medium     |
| Graph RAG       |     Very High |         Medium |           Very High |      🔴 High      |
| Multi-Stage RAG |        Medium |           High |              Medium |   🟡 Medium–High  |
| CRAG            |        Medium |           High |              Medium |      🔴 High      |
| Self-RAG        |        Medium |      Very High |              Medium |    🔴 Very High   |
| Adaptive RAG    |        Medium |         Medium |              Medium |     🟡 Medium     |
| Agentic RAG     |        Medium |      Very High |                High |    🔴 Very High   |
| RAPTOR          |     Very High |         Medium |              Medium |      🔴 High      |
| ColBERT         |          High |           High |                High |      🔴 High      |
| Multi-Agent     |          High | Extremely High |           Very High | 🔴 Extremely High |

--------------------------------------
## Typical LLM Calls Per User Query

| Architecture    | Typical LLM Calls |
| --------------- | ----------------: |
| Naive RAG       |                 1 |
| Advanced RAG    |               1–2 |
| Hybrid RAG      |                 1 |
| Graph RAG       |                 1 |
| Multi-Stage RAG |               1–2 |
| CRAG            |               2–3 |
| Self-RAG        |               3–8 |
| Adaptive RAG    |               1–2 |
| Agentic RAG     |             3–10+ |
| RAPTOR          |                 1 |
| ColBERT         |                 1 |
| Multi-Agent RAG |             5–20+ |
|
-------------------------------------
## Recommendation by Budget

| Budget                                   | Recommended Architectures                    |
| ---------------------------------------- | -------------------------------------------- |
| 💲 **Low Budget**                        | Naive RAG                                    |
| 💲💲 **Moderate Budget**                 | Advanced RAG, Hybrid RAG, Adaptive RAG       |
| 💲💲💲 **Higher Budget**                 | Multi-Stage RAG, Graph RAG, RAPTOR, ColBERT  |
| 💲💲💲💲 **Enterprise / Premium Budget** | CRAG, Self-RAG, Agentic RAG, Multi-Agent RAG |


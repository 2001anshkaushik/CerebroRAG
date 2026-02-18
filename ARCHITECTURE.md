# CerebroRAG Technical Whitepaper

## 1. System Overview
**CerebroRAG** is an autonomous retrieval-augmented generation engine designed to handle complex, ambiguous, or evolving queries. Unlike static RAG systems, CerebroRAG employs an **Agentic Loop** that actively reasons about retrieval quality and self-corrects when necessary.

## 2. Core Architecture

### 2.1 The Agentic Loop
The system is orchestrated by a central Agent that follows a strictly defined cognitive process:
1.  **Safety Evaluation**: All inputs pass through a strict regex-based Guardrails layer.
2.  **Intent Routing**: A lightweight classifier determines if the query needs external data (`SEARCH`) or logic/memory (`DIRECT`).
3.  **Retrieval & Verification**: If data is needed, it executes a Hybrid Search. Crucially, a "Judge" LLM validates the retrieved context. If relevance is low (< 0.6), the Agent triggers **Query Decomposition** to break the problem down effectively.

### 2.2 Hybrid Retrieval with Cross-Encoder Reranking
To solve the "Lost in the Middle" phenomenon, we implement a multi-stage funnel:
*   **Stage 1: Broad Recall**: Parallel execution of ChromaDB (Semantic) and BM25 (Keyword) searches for high recall.
*   **Stage 2: Ensemble**: Candidates are deduplicated and merged.
*   **Stage 3: Precision Reranking**: An LLM-based Cross-Encoder re-scores every candidate chunk specifically for the user's query, ensuring only the most relevant data hits the context window.

### 2.3 Semantic Memory
The system maintains long-term persistence without "context clutter":
*   **Fact Extraction**: After every turn, the Agent extracts high-signal facts (User preferences, Domain knowledge).
*   **Semantic Merging**: New facts are embedded and compared against existing memories. If a similar fact exists (Similarity > 0.85), an LLM merges them into a single coherent statement, preventing duplication.

## 3. Technology Stack
*   **Language**: Python 3.10+
*   **Orchestration**: LangChain
*   **Vector Database**: ChromaDB (Local Persistence)
*   **Embeddings**: OpenAI `text-embedding-3-small`
*   **LLM Engine**: GPT-4o-mini (Cost-optimized for high throughput)
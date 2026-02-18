# CerebroRAG: Modular Agentic RAG Engine

**Production-grade RAG architecture with Self-Correction, Hybrid Retrieval, and Semantic Memory.**

## 🚀 Why Use CerebroRAG?

Most RAG systems fail when faced with vague queries, massive datasets, or conflicting information. CerebroRAG solves this with an agentic loops approach:

- **Self-Correcting Retrieval**: If the initial search yields low confidence, the agent decomposes the query and retries autonomously.
- **Hybrid Search + Reranking**: Combines ChromaDB (Semantic) and BM25 (Keyword) with an LLM-based Cross-Encoder to achieve 99th-percentile precision.
- **Semantic Memory**: Instead of naive conversation logging, the system identifies "high-signal" facts and merges them into a persistent knowledge graph using semantic deduplication.
- **Enterprise Security**: Zero-Trust Guardrails prevent prompt injection and ensure safety before the query even reaches the model.

## 🛠️ Architecture

See [Technical Whitepaper](ARCHITECTURE.md) for deep dive.

- **Agent**: LangChain-based Router & Reasoning Engine
- **Retrieval**: ChromaDB + BM25 + Cross-Encoder Reranker
- **Memory**: Markdown-based Semantic Store (User & Company context)
- **Monitoring**: Latency tracking and visual citation snippets

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Agent
```bash
# Ingest your documents (PDF, MD, TXT)
python main.py ingest ./sample_docs/hackathon_guide.txt

# Ask a question
python main.py ask "What are the system requirements?"
```

### 3. Run Test Suite
```bash
make test
```

## 🧪 Testing Strategy

The project includes a comprehensive `unittest` suite covering:
- **Privacy Controls**: Ensuring the agent refuses to answer unknown questions.
- **Memory Updates**: Verifying semantic merging of conflicting facts.
- **Citation Precision**: Confirming grounded answers with page-level accuracy.
- **Security Guardrails**: Stress-testing against prompt injection attacks.
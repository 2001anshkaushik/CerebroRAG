import os
import pickle
import json
import time
import functools
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain Imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import Config

# Latency Decorator - Critical for monitoring RAG performance during demo
def measure_latency(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"[{func.__name__}] Latency: {latency_ms:.2f} ms")
        return result
    return wrapper

class HybridRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
        
        # Initialize Vector Store (Chroma) for Semantic Search capability
        self.vectorstore = Chroma(
            collection_name="agentic_rag",
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Initialize BM25 (Keyword) for lexical precision
        self.bm25_retriever = self._load_bm25()

    def _load_bm25(self) -> Optional[BM25Retriever]:
        """Loads pre-computed BM25 index to avoid re-indexing latency on startup."""
        if os.path.exists(Config.BM25_PERSIST_PATH):
            try:
                with open(Config.BM25_PERSIST_PATH, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load BM25 retriever: {e}")
        return None

    def _save_bm25(self):
        """Persists BM25 index to disk."""
        if self.bm25_retriever:
            os.makedirs(os.path.dirname(Config.BM25_PERSIST_PATH), exist_ok=True)
            with open(Config.BM25_PERSIST_PATH, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)

    def ingest_document(self, file_path: str):
        """
        Ingests a document into both Vector and Keyword stores.
        Handles PDF/TXT/MD parsing and preserves page metadata for citations.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Load Document based on extension
        loaded_docs = []
        if path_obj.suffix.lower() == '.pdf':
            try:
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
                return
        elif path_obj.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(file_path) if path_obj.suffix == '.txt' else UnstructuredMarkdownLoader(file_path)
            loaded_docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {path_obj.suffix}")

        if not loaded_docs:
            print(f"No content loaded from {file_path}")
            return

        # 2. Semantic Chunking Strategy
        # Using overlap to maintain context between splits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(loaded_docs)

        # 3. Enrich Metadata for Strict Citations
        for doc in split_docs:
            doc.metadata["source"] = path_obj.name
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            else:
                # Ensure 1-based indexing for PDFs (PyPDFLoader is 0-based)
                doc.metadata["page"] = doc.metadata["page"] + 1

        # 4. Update Vector Store
        self.vectorstore.add_documents(split_docs)

        # 5. Rebuild BM25 Index
        # We must rebuild the entire BM25 index as it relies on corpus statistics
        existing_data = self.vectorstore.get()
        if existing_data and existing_data['documents']:
             all_corpus_docs = []
             for i, text in enumerate(existing_data['documents']):
                 meta = existing_data['metadatas'][i] if existing_data['metadatas'] else {}
                 all_corpus_docs.append(Document(page_content=text, metadata=meta))
             
             self.bm25_retriever = BM25Retriever.from_documents(all_corpus_docs)
        else:
             self.bm25_retriever = BM25Retriever.from_documents(split_docs)
             
        self.bm25_retriever.k = Config.RETRIEVAL_K
        self._save_bm25()
        
        print(f"Ingested {len(split_docs)} chunks from {path_obj.name}")

    def rerank_chunks(self, query: str, retrieved_docs: List[Document]) -> List[Document]:
        """
        Perform Batch Reranking using an LLM as a Cross-Encoder.
        Optimized to handle multiple chunks in a single API call to reduce latency.
        """
        if not retrieved_docs:
            return []
            
        # Prepare Batch Input for LLM
        context_batch = []
        for i, doc in enumerate(retrieved_docs):
            # Truncate content slightly to fit multiple chunks in context window
            clean_content = doc.page_content[:500].replace("\n", " ") 
            context_batch.append(f"ID {i}: {clean_content}")
        
        joined_context = "\n".join(context_batch)
        
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of the following document chunks to the query: "{query}"
            
            Chunks:
            {chunks}
            
            Return ONLY a valid JSON object with a key 'scores' containing a list of objects {{'id': <int>, 'score': <float 0.0-1.0>}}.
            Example: {{"scores": [{{"id": 0, "score": 0.9}}, {{"id": 1, "score": 0.1}}]}}
            """
        )
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"query": query, "chunks": joined_context})
            scores_map = {item['id']: float(item['score']) for item in result.get('scores', [])}
            
            # Apply relevance scores to metadata
            scored_docs = []
            for i, doc in enumerate(retrieved_docs):
                score = scores_map.get(i, 0.0)
                doc.metadata["relevance_score"] = score
                scored_docs.append(doc)
                
            # Sort by Relevance Score (High to Low)
            scored_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
            return scored_docs[:Config.RERANK_TOP_N]
            
        except Exception as e:
            print(f"Batch Reranking failed ({e}). Falling back to original retrieval order.")
            return retrieved_docs[:Config.RERANK_TOP_N]

    @measure_latency
    def search(self, query: str) -> List[Document]:
        """
        Executes the Hybrid Retrieval Pipeline:
        1. Semantic Search (Chroma)
        2. Keyword Search (BM25)
        3. Ensemble (Union + Deduplication)
        4. Cross-Encoder Reranking (LLM)
        """
        # 1. Semantic Vector Search
        semantic_docs = self.vectorstore.similarity_search(query, k=Config.RETRIEVAL_K)
        
        # 2. Lexical Keyword Search
        lexical_docs = []
        if self.bm25_retriever:
            try:
                lexical_docs = self.bm25_retriever.get_relevant_documents(query)
            except Exception:
                lexical_docs = []
            
        # 3. Ensemble Strategy (Union + Deduplication)
        # We manually deduplicate based on content hash to ensure diverse results
        unique_content_set = set()
        ensemble_candidates = []
        
        for doc in semantic_docs + lexical_docs:
            # Create a signature based on content location
            doc_signature = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
            if doc_signature not in unique_content_set:
                unique_content_set.add(doc_signature)
                ensemble_candidates.append(doc)
        
        if not ensemble_candidates:
            return []
            
        print(f"Reranking {len(ensemble_candidates)} candidate chunks...")
        
        # 4. LLM-Based Reranking for Precision
        final_ranked_docs = self.rerank_chunks(query, ensemble_candidates)
        
        return final_ranked_docs

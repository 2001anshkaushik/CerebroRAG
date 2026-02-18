import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY") 

    # Models
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Paths - Adjusted for CerebroRAG/src/config.py
    # Base dir is the parent of 'src', i.e., 'CerebroRAG'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data Paths
    USER_MEMORY_PATH = os.path.join(BASE_DIR, "data", "USER_MEMORY.md")
    COMPANY_MEMORY_PATH = os.path.join(BASE_DIR, "data", "COMPANY_MEMORY.md")
    
    # DB Paths
    CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR, "db", "chroma_db")
    BM25_PERSIST_PATH = os.path.join(BASE_DIR, "db", "bm25_retriever.pkl")
    
    # Retrieval
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 5
    RERANK_TOP_N = 3
    
    # Thresholds
    RELEVANCE_THRESHOLD = 0.6
    MEMORY_DEDUPE_THRESHOLD = 0.85 

    @staticmethod
    def validate():
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing via .env or environment variable.")

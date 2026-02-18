import os
import numpy as np
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config

class MemoryManager:
    """
    Manages semantic long-term memory for the Agent.
    Handles storage, cosine similarity checks, and LLM-based deduplication/merging.
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)

    def _get_file_path(self, category: str) -> str:
        """Determines the correct memory store based on category intent."""
        if category.upper() == 'USER':
            return Config.USER_MEMORY_PATH
        elif category.upper() == 'COMPANY':
            return Config.COMPANY_MEMORY_PATH
        else:
            raise ValueError("Category must be USER or COMPANY")

    def _read_memories(self, file_path: str) -> List[str]:
        """Reads existing high-signal facts from Markdown storage."""
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        facts = []
        for line in lines:
            line = line.strip()
            # Only extract bullet points to avoid parsing headers/comments
            if line.startswith('- '):
                facts.append(line[2:].strip())
        return facts

    def _write_memories(self, file_path: str, facts: List[str]):
        """Persists memories to disk with a clean Markdown header."""
        # Ensure directory structure exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        header = ""
        if "USER_MEMORY" in file_path:
             header = "# USER MEMORY\n\n<!--\nAppend only high-signal, user-specific facts worth remembering.\nDo NOT dump raw conversation.\nAvoid secrets or sensitive information.\n-->\n\n"
        else:
             header = "# COMPANY MEMORY\n\n<!--\nAppend reusable org-wide learnings that could help colleagues too.\nDo NOT dump raw conversation.\nAvoid secrets or sensitive information.\n-->\n\n"
             
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            for fact in facts:
                f.write(f"- {fact}\n")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates Semantic Similarity between two embedding vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _merge_facts(self, existing_fact: str, new_fact: str) -> str:
        """
        Uses an LLM to smartly merge two similar facts into one concise statement
        to prevent memory bloat (e.g., 'User is Alice' + 'Alice is a dev' -> 'User is Alice, a dev').
        """
        prompt = ChatPromptTemplate.from_template(
            """Merge the following two facts into a single, concise, high-signal fact.
            Ensure no information is lost, but redundancy is removed.
            
            Fact 1: {fact1}
            Fact 2: {fact2}
            
            Merged Fact:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        merged_output = chain.invoke({"fact1": existing_fact, "fact2": new_fact})
        return merged_output.strip()

    def write_memory(self, new_fact: str, category: str = 'USER'):
        """
        Public API to write a memory.
        Performs Semantic Deduplication check before writing.
        """
        target_file_path = self._get_file_path(category)
        existing_facts = self._read_memories(target_file_path)
        
        # Initial Case: No memory yet
        if not existing_facts:
            self._write_memories(target_file_path, [new_fact])
            print(f"Memory: Appended new fact to {category}.")
            return

        # Semantic Search against existing memories
        new_fact_embedding = self.embeddings.embed_query(new_fact)
        existing_embeddings = self.embeddings.embed_documents(existing_facts)
        
        highest_similarity = -1.0
        most_similar_idx = -1
        
        for i, emb in enumerate(existing_embeddings):
            sim = self._cosine_similarity(new_fact_embedding, emb)
            if sim > highest_similarity:
                highest_similarity = sim
                most_similar_idx = i
                
        # Deduplication Logic
        if highest_similarity > Config.MEMORY_DEDUPE_THRESHOLD:
            print(f"Memory: Found similar fact (Score: {highest_similarity:.2f}). Merging...")
            merged_fact = self._merge_facts(existing_facts[most_similar_idx], new_fact)
            existing_facts[most_similar_idx] = merged_fact
            self._write_memories(target_file_path, existing_facts)
            print(f"Memory: Merged '{new_fact}' into existing memory.")
        else:
            existing_facts.append(new_fact)
            self._write_memories(target_file_path, existing_facts)
            print(f"Memory: Appended new fact to {category}.")

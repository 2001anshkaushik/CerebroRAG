import json
import re
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.config import Config
from src.retriever import HybridRetriever
from src.memory_manager import MemoryManager
from src.guardrails import Guardrails

class Agent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.memory_manager = MemoryManager()
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.7)
        self.judge_llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)

    def _router_decision(self, query: str) -> str:
        """
        Decision Step: Determines if the query requires external knowledge (RAG)
        or if it's a general conversation/memory recall to avoid unnecessary computation.
        
        Returns: 'SEARCH' or 'DIRECT'
        """
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """Determine the best strategy to answer the user query.
            
            Query: {query}
            
            Strategy Definitions:
            - SEARCH: Use this if the query requires factual knowledge, documentation, specific data, or details about the company/project (e.g. "What are the rules?", "Project status").
            - DIRECT: Use this if the query is a greeting, a compliment, or a request to summarize previous conversation (which you have in context), or general chitchat (e.g. "Hi", "Who are you?").
            
            Return ONLY a valid JSON object with a key 'decision' equal to "SEARCH" or "DIRECT".
            """
        )
        chain = prompt | self.judge_llm | parser
        try:
            result = chain.invoke({"query": query})
            decision = result.get("decision", "SEARCH").upper()
            return decision if decision in ["SEARCH", "DIRECT"] else "SEARCH"
        except:
            # Fallback to SEARCH to be safe if decision fails
            return "SEARCH"

    def _judge_relevance(self, query: str, context: str) -> float:
        """
        Self-Correction Mechanism: Evaluates if the retrieved context is actually relevant.
        Used to trigger query decomposition if the search results are poor.
        """
        if not context:
            return 0.0

        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """You are a strict relevance judge.
            Query: {query}
            
            Retrieved Context:
            {context}
            
            Is this context sufficient to answer the query with high confidence?
            Return ONLY a valid JSON object with 'score' (0.0 to 1.0) and 'reasoning'.
            Example: {{"score": 0.4, "reasoning": "Missing specific details about X"}}
            """
        )
        chain = prompt | self.judge_llm | parser
        try:
            result = chain.invoke({"query": query, "context": context[:2000]})
            return float(result.get("score", 0.0))
        except Exception:
            return 0.5 # Conservative fallback

    def _decompose_query(self, query: str) -> List[str]:
        """
        Breaks down complex or vague queries into keyword-centric sub-queries
        to improve retrieval coverage when initial search fails.
        """
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """Break the following query into 2-3 simpler, keyword-centric search queries that would help answer it.
            Query: {query}
            
            Return ONLY a valid JSON object with a key 'queries' containing a list of strings.
            Example: {{"queries": ["population of france", "gdp of france 2023"]}}
            """
        )
        chain = prompt | self.judge_llm | parser
        try:
            result = chain.invoke({"query": query})
            return result.get("queries", [query])
        except:
            return [query]

    def _extract_memories_async(self, query: str, answer: str):
        """
        Post-Process: Extracts high-signal facts to update long-term knowledge graphs (User/Company Memory).
        This runs 'async' conceptually (in this demo, it's consistent).
        """
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """Analyze the interaction and extract high-signal facts about the USER or the COMPANY/DOMAIN.
            Ignore general knowledge (e.g., "The sky is blue").
            Focus on user preferences, specific project details, or organizational logic.
            
            User Query: {query}
            AI Answer: {answer}
            
            Return a valid JSON object with two lists: 'user_facts' and 'company_facts'.
            Example: {{"user_facts": ["User is a finance analyst"], "company_facts": ["Project X uses Python"]}}
            If no facts, return empty lists.
            """
        )
        chain = prompt | self.judge_llm | parser
        try:
            result = chain.invoke({"query": query, "answer": answer})
            for fact in result.get("user_facts", []):
                self.memory_manager.write_memory(fact, "USER")
            for fact in result.get("company_facts", []):
                self.memory_manager.write_memory(fact, "COMPANY")
        except Exception as e:
            pass 

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main Agentic Loop:
        1. Guardrails -> 2. Router -> 3. Retrieval (if needed) -> 4. Self-Correction -> 5. Generation -> 6. Memory
        """
        print(f"Agent: Processing query '{query}'")
        
        # 1. Guardrails Check (Zero-Trust Security)
        if not Guardrails.validate_query(query):
            return {
                "question": query,
                "answer": "I cannot fulfill this request as it violates safety policies.",
                "citations": []
            }

        # 2. Reasoning Step (Router)
        decision = self._router_decision(query)
        print(f"Agent: Router Decision = {decision}")
        
        retrieved_docs = []
        context_text = ""
        
        if decision == "SEARCH":
            # 3. Initial Semantic + Keyword Search
            retrieved_docs = self.retriever.search(query)
            context_text = "\n\n".join([f"Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})\nContent: {d.page_content}" for d in retrieved_docs])
            
            # 4. Self-Correction Loop
            # If the initial context is poor, we decompose the query and try again.
            relevance_score = self._judge_relevance(query, context_text)
            print(f"Agent: Relevance Score = {relevance_score:.2f}")
            
            if relevance_score < Config.RELEVANCE_THRESHOLD:
                print("Agent: Low relevance. Triggering Query Decomposition.")
                sub_queries = self._decompose_query(query)
                print(f"Agent: Sub-queries: {sub_queries}")
                
                all_candidate_docs = []
                for sub_q in sub_queries:
                    new_docs = self.retriever.search(sub_q)
                    all_candidate_docs.extend(new_docs)
                
                # Deduplicate results from sub-queries
                seen_content = set()
                unique_docs = []
                for d in all_candidate_docs:
                    if d.page_content not in seen_content:
                        seen_content.add(d.page_content)
                        unique_docs.append(d)
                
                if unique_docs:
                    # Re-rank the expanded set of documents
                    retrieved_docs = self.retriever.rerank_chunks(query, unique_docs) 
                    context_text = "\n\n".join([f"Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})\nContent: {d.page_content}" for d in retrieved_docs])
        
        else:
            print("Agent: Skipping retrieval (Direct/General query).")

        # 5. Generation with Citations
        system_prompt = """You are a helpful RAG Assistant.
        Answer the user's question using ONLY the provided context if available.
         If no context is provided (DIRECT mode), answer based on your general knowledge and conversation history (if any), but do not hallucinate external docs.
        
        CRITICAL CITATION RULE:
        If you use information from the Context, you MUST cite it.
        Format: [[Source: <filename>, Page: <page_number>]].
        Example: "The revenue was $5M [[Source: financial_report.pdf, Page: 12]]."
        
        If the Context does not contain the answer and you are in SEARCH mode, say "I cannot find the answer in the provided documents."
        """
        
        chain_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        gen_chain = chain_prompt | self.llm | StrOutputParser()
        final_answer = gen_chain.invoke({"context": context_text, "query": query})
        
        # 6. Post-Process Output & Strict Citation Filtering
        # We only return citations that were actually referenced in the answer text.
        active_citations = []
        
        # Extract cited sources from the answer text to ensure precision
        # Pattern matches "Source: <filename>" inside the brackets
        cited_files_in_answer = set(re.findall(r"Source:\s*([^,\]]+)", final_answer))
        
        for d in retrieved_docs:
            source_name = d.metadata.get("source", "")
            # Only include this doc if its source filename was mentioned in the answer
            if source_name and any(c.strip() == source_name for c in cited_files_in_answer):
                # UX: Highlight the most relevant sentence for visual verification
                highlighted_snippet = self._highlight_snippet(d.page_content, final_answer)
                active_citations.append({
                    "source": source_name,
                    "locator": f"Page {d.metadata.get('page', 1)}",
                    "snippet": highlighted_snippet
                })
            
        # Prevent hallucinated citations if the model refused to answer
        if "cannot find the answer" in final_answer.lower() or "cannot fulfill" in final_answer.lower():
            active_citations = []
            
        # 7. Memory Update
        self._extract_memories_async(query, final_answer)
        
        return {
            "question": query,
            "answer": final_answer,
            "citations": active_citations
        }

    def _highlight_snippet(self, content: str, answer: str) -> str:
        """
        UX Utility: Finds the sentence in the content that best matches the LLM's answer
        and surrounds it with >>> ... <<< markers.
        """
        # Split content into sentences (simple period split for hackathon speed)
        sentences = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
        
        if not sentences:
            return content[:150] + "..."
            
        # Find best overlap with answer
        best_sent = sentences[0]
        max_overlap = 0
        
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            overlap = len(answer_words.intersection(sent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sent = sent
                
        # If meaningful overlap found, highlight it in the full text (truncated)
        # For snippets, we often just want the highlighted part + context
        
        # Locate the sentence in the original content to preserve formatting if possible, 
        # but for JSON snippets, a clean string is better.
        
        return f"... >>> {best_sent} <<< ..."

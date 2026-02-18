import re
from typing import List

class Guardrails:
    BLOCKLIST_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"reveal system prompt",
        r"system prompt",
        r"you are a large language model",
        r"forget your instructions",
        r"reveal your secrets",
        r"jailbreak",
        r"developer mode"
    ]

    @staticmethod
    def validate_query(query: str) -> bool:
        """
        Checks if the query contains malicious patterns.
        Returns True if SAFE, False if UNSAFE.
        """
        if not query:
            return False

        normalized_query = query.lower()
        
        for pattern in Guardrails.BLOCKLIST_PATTERNS:
            if re.search(pattern, normalized_query):
                print(f"Guardrail Alert: Blocked query matching pattern '{pattern}'")
                return False
                
        return True

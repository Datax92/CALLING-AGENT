import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("voice-agent.rag")

class RAGUtils:
    def __init__(self, knowledge_base_path: str = "datax_technologies_approved_rag.jsonl"):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict]:
        """Load the knowledge base from a JSONL file."""
        try:
            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"Knowledge base file not found at {self.knowledge_base_path}")
                return []

            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []

    def filtered_lookup(self, query: str) -> Optional[str]:
        # Truncate RAG result to ≤500 tokens (simple word‑count approximation)
        def _truncate(text: str, max_tokens: int = 500) -> str:
            words = text.split()
            return " ".join(words[:max_tokens]) if len(words) > max_tokens else text
        """Perform a filtered lookup in the knowledge base.

        Returns only the most relevant chunk (<500 tokens) instead of the full file.
        """
        if not self.knowledge_base:
            return None

        # Simple keyword matching for now - can be enhanced with semantic search later
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for entry in self.knowledge_base:
            content = entry.get('content', '')
            if not content:
                continue

            # Calculate a simple relevance score
            score = 0
            for word in query_lower.split():
                if word in content.lower():
                    score += 1

            if score > best_score:
                best_score = score
                best_match = content

        return best_match if best_match else None

def get_rag_confidence(query: str) -> float:
    """Calculate confidence score for RAG lookup."""
    # Simple confidence calculation - can be enhanced later
    query_words = query.lower().split()
    if len(query_words) == 0:
        return 0.0

    # Default confidence for now
    return 0.1
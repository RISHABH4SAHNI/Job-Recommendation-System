"""Fuzzy string matching utilities"""
from fuzzywuzzy import fuzz
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class FuzzyMatcher:
    """Fuzzy string matching for skills and text comparison"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using fuzzy matching"""
        if not str1 or not str2:
            return 0.0

        # Use token sort ratio for better matching of multi-word terms
        similarity = fuzz.token_sort_ratio(str1.lower(), str2.lower()) / 100.0
        return similarity

    def find_best_matches(self, query: str, candidates: List[str], limit: int = 5) -> List[Tuple[str, float]]:
        """Find best matching candidates for a query string"""
        matches = []

        for candidate in candidates:
            similarity = self.calculate_similarity(query, candidate)
            matches.append((candidate, similarity))

        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:limit]

    def is_similar(self, str1: str, str2: str) -> bool:
        """Check if two strings are similar based on threshold"""
        similarity = self.calculate_similarity(str1, str2)
        return similarity >= self.threshold

    def deduplicate_list(self, items: List[str]) -> List[str]:
        """Remove similar duplicates from a list of strings"""
        unique_items = []

        for item in items:
            if not any(self.is_similar(item, unique_item) for unique_item in unique_items):
                unique_items.append(item)

        return unique_items

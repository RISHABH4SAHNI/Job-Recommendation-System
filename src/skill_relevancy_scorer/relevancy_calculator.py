"""Relevancy score calculation between candidates and jobs"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .skill_extractor import SkillExtractor
from .fuzzy_matcher import FuzzyMatcher

logger = logging.getLogger(__name__)

class RelevancyCalculator:
    """Calculate relevancy scores between candidates and jobs"""

    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.fuzzy_matcher = FuzzyMatcher()
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    def calculate_skill_match_score(self, candidate_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill match score using fuzzy matching"""
        if not candidate_skills or not job_skills:
            return 0.0

        total_score = 0.0
        max_possible_score = len(job_skills)

        for job_skill in job_skills:
            best_match_score = 0.0
            for candidate_skill in candidate_skills:
                similarity = self.fuzzy_matcher.calculate_similarity(candidate_skill, job_skill)
                best_match_score = max(best_match_score, similarity)
            total_score += best_match_score

        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    def calculate_text_similarity(self, candidate_text: str, job_text: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        try:
            # Combine texts for vectorization
            texts = [candidate_text, job_text]

            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            return similarity_matrix[0][0]
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0

    def calculate_domain_match_score(self, candidate_domains: List[str], job_domain: str) -> float:
        """Calculate domain match score"""
        if not candidate_domains or not job_domain:
            return 0.0

        # Direct match
        if job_domain.lower() in [domain.lower() for domain in candidate_domains]:
            return 1.0

        # Fuzzy match with domains
        best_score = 0.0
        for candidate_domain in candidate_domains:
            similarity = self.fuzzy_matcher.calculate_similarity(candidate_domain, job_domain)
            best_score = max(best_score, similarity)

        return best_score

    def calculate_comprehensive_relevancy(self, candidate_row: Dict[str, Any], job_row: Dict[str, Any]) -> float:
        """Calculate comprehensive relevancy score between candidate and job"""

        # Extract skills
        candidate_skills = self.skill_extractor.extract_skills_from_candidate(candidate_row)
        job_skills = self.skill_extractor.extract_skills_from_job(job_row)

        # Calculate skill match score (weight: 0.4)
        skill_score = self.calculate_skill_match_score(candidate_skills, job_skills)

        # Prepare text for similarity calculation
        candidate_text = self._prepare_candidate_text(candidate_row)
        job_text = self._prepare_job_text(job_row)

        # Calculate text similarity (weight: 0.3)
        text_similarity = self.calculate_text_similarity(candidate_text, job_text)

        # Calculate domain match (weight: 0.3)
        candidate_domains = self._extract_candidate_domains(candidate_row)
        job_domain = job_row.get('domain', 'Other')
        domain_score = self.calculate_domain_match_score(candidate_domains, job_domain)

        # Weighted final score
        final_score = (0.4 * skill_score) + (0.3 * text_similarity) + (0.3 * domain_score)

        return final_score

    def _prepare_candidate_text(self, candidate_row: Dict[str, Any]) -> str:
        """Prepare candidate text for similarity calculation"""
        text_parts = []
        text_fields = ['Experiences', 'Projects', 'Achievements', 'Major']

        for field in text_fields:
            if field in candidate_row and candidate_row[field]:
                text_parts.append(str(candidate_row[field]))

        return " ".join(text_parts)

    def _prepare_job_text(self, job_row: Dict[str, Any]) -> str:
        """Prepare job text for similarity calculation"""
        text_parts = []
        text_fields = ['role_description', 'requirement', 'description', 'role_title']

        for field in text_fields:
            if field in job_row and job_row[field]:
                text_parts.append(str(job_row[field]))

        return " ".join(text_parts)

    def _extract_candidate_domains(self, candidate_row: Dict[str, Any]) -> List[str]:
        """Extract domains from candidate profile"""
        domains = []

        if 'RecommendedJobDomains' in candidate_row and candidate_row['RecommendedJobDomains']:
            try:
                import json
                if isinstance(candidate_row['RecommendedJobDomains'], str):
                    domain_data = json.loads(candidate_row['RecommendedJobDomains'])
                    for domain_info in domain_data:
                        if 'job_domain' in domain_info:
                            domains.append(domain_info['job_domain'])
            except:
                pass

        # Also consider major as potential domain
        if 'Major' in candidate_row and candidate_row['Major']:
            domains.append(candidate_row['Major'])

        return domains

"""Tests for matching engine"""
import unittest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from skill_relevancy_scorer import RelevancyCalculator
from matching_engine import CandidateJobMatcher

class TestRelevancyCalculator(unittest.TestCase):
    """Test relevancy calculation"""

    def setUp(self):
        self.calculator = RelevancyCalculator()

    def test_calculate_skill_match_score(self):
        """Test skill matching score calculation"""
        candidate_skills = ['python', 'machine learning', 'sql']
        job_skills = ['python', 'data analysis', 'statistics']

        score = self.calculator.calculate_skill_match_score(candidate_skills, job_skills)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_calculate_text_similarity(self):
        """Test text similarity calculation"""
        candidate_text = "Experienced Python developer with machine learning background"
        job_text = "Looking for Python developer with ML experience"

        similarity = self.calculator.calculate_text_similarity(candidate_text, job_text)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

class TestCandidateJobMatcher(unittest.TestCase):
    """Test candidate job matching"""

    def setUp(self):
        self.matcher = CandidateJobMatcher()

    def test_match_candidates_to_jobs(self):
        """Test matching candidates to jobs"""
        # Sample candidate data
        candidates_df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith'],
            'Email': ['john@email.com', 'jane@email.com'],
            'HardSkills': [
                json.dumps([{'skill': 'python', 'percentage': 90}]),
                json.dumps([{'skill': 'marketing', 'percentage': 85}])
            ],
            'Experiences': ['Python developer', 'Marketing manager'],
            'RecommendedJobDomains': [
                json.dumps([{'job_domain': 'Software Development'}]),
                json.dumps([{'job_domain': 'Marketing'}])
            ]
        })

        # Sample job data
        jobs_df = pd.DataFrame({
            'job_id': ['1', '2'],
            'role_title': ['Python Developer', 'Marketing Specialist'],
            'role_description': ['Python programming', 'Digital marketing'],
            'company_name': ['Tech Corp', 'Marketing Inc'],
            'domain': ['Software Development', 'Marketing']
        })

        matches = self.matcher.match_candidates_to_jobs(candidates_df, jobs_df, top_k=5)
        self.assertIsInstance(matches, pd.DataFrame)
        self.assertIn('JobRole', matches.columns)
        self.assertIn('RelevantCandidates', matches.columns)

if __name__ == '__main__':
    unittest.main()
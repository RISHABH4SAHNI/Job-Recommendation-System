"""Tests for job domain classifier"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from job_domain_classifier import JobDomainClassifier
from job_domain_classifier.preprocessing import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing functionality"""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text(self):
        """Test text cleaning"""
        text = "We are looking for a Python developer with experience in Django!"
        cleaned = self.preprocessor.clean_text(text)

        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!", cleaned)
        self.assertIn("python", cleaned.lower())

    def test_assign_domain(self):
        """Test domain assignment"""
        text = "We need a Python developer with machine learning experience"
        domain = self.preprocessor.assign_domain(text)

        self.assertIn(domain, ['Software Development', 'Data Science', 'Other'])

    def test_preprocess_job_data(self):
        """Test job data preprocessing"""
        sample_data = pd.DataFrame({
            'role_description': ['Python developer needed', 'Data scientist position'],
            'job_id': ['1', '2']
        })

        processed = self.preprocessor.preprocess_job_data(sample_data)

        self.assertIn('domain', processed.columns)
        self.assertIn('cleaned_text', processed.columns)
        self.assertEqual(len(processed), 2)

class TestJobDomainClassifier(unittest.TestCase):
    """Test job domain classifier"""

    def setUp(self):
        self.classifier = JobDomainClassifier()

    def test_prepare_data(self):
        """Test data preparation"""
        sample_data = pd.DataFrame({
            'role_description': ['Python developer', 'Marketing manager', 'Data scientist'],
            'job_id': ['1', '2', '3']
        })

        X, y = self.classifier.prepare_data(sample_data)

        self.assertIsInstance(X, (np.ndarray, type(X)))
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))

if __name__ == '__main__':
    unittest.main()
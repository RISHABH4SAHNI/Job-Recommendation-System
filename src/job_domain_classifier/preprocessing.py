"""Text preprocessing utilities for job domain classification"""
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing for job descriptions"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Job domain mappings extracted from the notebook
        self.job_domains = {
            "Software Development": [
                "android", "backend", "full stack", "node.js", "python", "web developer",
                "elixir", "phoenix", "sde", "react native", "software developer", "java", 
                "kotlin", "jetpack compose", "sdk", "firebase"
            ],
            "Data Science": [
                "data analyst", "data scientist", "big data", "machine learning",
                "data analytics", "prompt engineer", "mlops", "data analysis",
                "ai", "artificial intelligence", "statistical modeling", "deep learning"
            ],
            "Marketing": [
                "marketing", "brand marketing", "digital marketing", "social media",
                "influencer", "content creation", "seo", "email marketing",
                "product marketing", "advertising", "market research"
            ],
            "Human Resources": [
                "human resource", "hr", "recruitment", "talent acquisition",
                "comp & benefits", "employee relations", "training", "development"
            ],
            "Sales": [
                "sales", "business development", "inside sales", "account manager",
                "lead generation", "sales executive", "territory manager"
            ],
            "Operations": [
                "operations", "business operations", "supply chain", "logistics",
                "inventory management", "procurement", "project management"
            ],
            "Research": [
                "research", "insights", "data analysis", "market research",
                "academic research", "clinical research", "r&d"
            ],
            "Product Management": [
                "product management", "product solution", "product architect",
                "product owner", "product strategy", "product development"
            ],
            "Engineering": [
                "robotics", "unity", "climate", "ai", "automotive",
                "mechanical", "steering", "suspension", "brakes",
                "civil engineering", "electrical engineering", "chemical engineering"
            ]
        }

    def assign_domain(self, text: str) -> str:
        """Assign job domain based on keywords in text"""
        text = text.lower()
        for domain, skills in self.job_domains.items():
            if any(skill in text for skill in skills):
                return domain
        return "Other"

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            text = str(text).lower()
            # Remove non-alphanumeric characters except spaces
            text = "".join([char for char in text if char.isalnum() or char in " "])
            # Tokenize and remove stopwords
            words = [
                self.lemmatizer.lemmatize(word) 
                for word in text.split() 
                if word not in self.stop_words
            ]
            return " ".join(words)
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""

    def preprocess_job_data(self, df: pd.DataFrame, text_column: str = "role_description") -> pd.DataFrame:
        """Preprocess job data DataFrame"""
        logger.info(f"Preprocessing {len(df)} job records")

        # Drop rows with missing text
        df = df.dropna(subset=[text_column]).copy()

        # Assign domains
        df["domain"] = df[text_column].apply(self.assign_domain)

        # Clean text
        df["cleaned_text"] = df[text_column].apply(self.clean_text)

        # Remove empty cleaned text
        df = df[df["cleaned_text"].str.strip() != ""]

        logger.info(f"Preprocessed data: {len(df)} records, {df['domain'].nunique()} unique domains")
        return df
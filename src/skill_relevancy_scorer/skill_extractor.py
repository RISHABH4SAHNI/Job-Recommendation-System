"""Skill extraction utilities"""
import pandas as pd
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SkillExtractor:
    """Extract skills from job descriptions and candidate profiles"""

    def __init__(self):
        # Comprehensive skills list extracted from the notebook
        self.skills_list = [
            "python", "java", "kotlin", "jetpack compose", "android sdk", "firebase",
            "rest", "json", "proto", "sql", "javascript", "cloud computing", "aws",
            "excel", "data visualization", "react", "node.js", "marketing", "social media",
            "seo", "content creation", "product management", "sales", "business development",
            "hr", "research", "operations", "analytical skills", "problem solving",
            "communication", "collaboration", "organizational skills", "multitasking",
            "microsoft office", "ai", "machine learning", "big data", "deep learning",
            "neural networks", "statistical analysis", "pandas", "numpy", "scikit-learn",
            "tensorflow", "keras", "r", "sas", "sql", "tableau", "power bi",
            "lead generation", "b2b", "b2c", "market research", "product marketing",
            "email marketing", "content strategy", "creative writing", "employee engagement",
            "talent management", "recruitment", "project management", "agile", "scrum",
            "supply chain management", "logistics", "procurement", "inventory management"
        ]

        # Convert to lowercase for matching
        self.skills_set = set(skill.lower() for skill in self.skills_list)

    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from given text"""
        if not text or pd.isna(text):
            return []

        text = str(text).lower()
        found_skills = []

        for skill in self.skills_set:
            if skill in text:
                found_skills.append(skill)

        return found_skills

    def extract_skills_from_job(self, job_row: Dict[str, Any]) -> List[str]:
        """Extract skills from job description"""
        text_fields = ['role_description', 'requirement', 'description']
        all_skills = set()

        for field in text_fields:
            if field in job_row and job_row[field]:
                skills = self.extract_skills_from_text(job_row[field])
                all_skills.update(skills)

        return list(all_skills)

    def extract_skills_from_candidate(self, candidate_row: Dict[str, Any]) -> List[str]:
        """Extract skills from candidate profile"""
        all_skills = set()

        # Extract from hard skills if available
        if 'HardSkills' in candidate_row and candidate_row['HardSkills']:
            try:
                # Handle JSON string format
                import json
                if isinstance(candidate_row['HardSkills'], str):
                    hard_skills_data = json.loads(candidate_row['HardSkills'])
                    for skill_info in hard_skills_data:
                        if 'skill' in skill_info:
                            skill_name = skill_info['skill'].lower()
                            if skill_name in self.skills_set:
                                all_skills.add(skill_name)
            except:
                pass

        # Extract from experiences and projects text
        text_fields = ['Experiences', 'Projects', 'Achievements']
        for field in text_fields:
            if field in candidate_row and candidate_row[field]:
                skills = self.extract_skills_from_text(str(candidate_row[field]))
                all_skills.update(skills)

        return list(all_skills)

    def get_skill_categories(self) -> Dict[str, List[str]]:
        """Get skills organized by categories (based on job domains)"""
        categories = {
            "programming": [
                "python", "java", "kotlin", "javascript", "r", "sql", "c++", "c programming"
            ],
            "data_science": [
                "machine learning", "deep learning", "ai", "data analysis", "statistical analysis",
                "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "tableau", "power bi"
            ],
            "web_development": [
                "react", "node.js", "javascript", "html", "css", "rest", "json"
            ],
            "mobile_development": [
                "android sdk", "kotlin", "jetpack compose", "firebase", "react native"
            ],
            "business": [
                "marketing", "sales", "business development", "product management", "project management"
            ],
            "analytics": [
                "excel", "data visualization", "analytical skills", "business intelligence"
            ]
        }

        return categories
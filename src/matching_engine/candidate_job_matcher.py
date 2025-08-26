"""Candidate-Job matching logic"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json
import logging

from ..skill_relevancy_scorer import RelevancyCalculator

logger = logging.getLogger(__name__)

class CandidateJobMatcher:
    """Match candidates with relevant jobs"""

    def __init__(self):
        self.relevancy_calculator = RelevancyCalculator()

    def match_candidates_to_jobs(self, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame, 
                                top_k: int = 10) -> pd.DataFrame:
        """Match candidates to jobs and return relevancy scores"""
        logger.info(f"Matching {len(candidates_df)} candidates to {len(jobs_df)} jobs")

        matches = []

        for _, job_row in jobs_df.iterrows():
            job_matches = []
            job_dict = job_row.to_dict()

            for _, candidate_row in candidates_df.iterrows():
                candidate_dict = candidate_row.to_dict()

                # Calculate relevancy score
                relevancy_score = self.relevancy_calculator.calculate_comprehensive_relevancy(
                    candidate_dict, job_dict
                )

                job_matches.append({
                    'CandidateName': candidate_dict.get('Name', 'Unknown'),
                    'CandidateEmail': candidate_dict.get('Email', ''),
                    'RelevancyScore': relevancy_score
                })

            # Sort by relevancy score and take top k
            job_matches.sort(key=lambda x: x['RelevancyScore'], reverse=True)
            top_matches = job_matches[:top_k]

            matches.append({
                'JobRole': job_dict.get('role_title', 'Unknown Role'),
                'JobId': job_dict.get('job_id', ''),
                'Company': job_dict.get('company_name', ''),
                'RelevantCandidates': json.dumps(top_matches)
            })

        logger.info(f"Generated matches for {len(matches)} jobs")
        return pd.DataFrame(matches)

    def match_jobs_to_candidates(self, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame,
                                top_k: int = 10) -> pd.DataFrame:
        """Match jobs to candidates and return recommendations"""
        logger.info(f"Matching {len(jobs_df)} jobs to {len(candidates_df)} candidates")

        matches = []

        for _, candidate_row in candidates_df.iterrows():
            candidate_matches = []
            candidate_dict = candidate_row.to_dict()

            for _, job_row in jobs_df.iterrows():
                job_dict = job_row.to_dict()

                # Calculate relevancy score
                relevancy_score = self.relevancy_calculator.calculate_comprehensive_relevancy(
                    candidate_dict, job_dict
                )

                candidate_matches.append({
                    'JobRole': job_dict.get('role_title', 'Unknown Role'),
                    'JobId': job_dict.get('job_id', ''),
                    'Company': job_dict.get('company_name', ''),
                    'Location': job_dict.get('location', ''),
                    'Stipend': job_dict.get('stipend', ''),
                    'RelevancyScore': relevancy_score
                })

            # Sort by relevancy score and take top k
            candidate_matches.sort(key=lambda x: x['RelevancyScore'], reverse=True)
            top_matches = candidate_matches[:top_k]

            matches.append({
                'CandidateName': candidate_dict.get('Name', 'Unknown'),
                'CandidateEmail': candidate_dict.get('Email', ''),
                'RecommendedJobs': json.dumps(top_matches)
            })

        logger.info(f"Generated recommendations for {len(matches)} candidates")
        return pd.DataFrame(matches)

    def generate_detailed_match_report(self, candidate_dict: Dict[str, Any], 
                                     job_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed match report for a candidate-job pair"""

        # Calculate individual scores
        candidate_skills = self.relevancy_calculator.skill_extractor.extract_skills_from_candidate(candidate_dict)
        job_skills = self.relevancy_calculator.skill_extractor.extract_skills_from_job(job_dict)

        skill_score = self.relevancy_calculator.calculate_skill_match_score(candidate_skills, job_skills)

        candidate_text = self.relevancy_calculator._prepare_candidate_text(candidate_dict)
        job_text = self.relevancy_calculator._prepare_job_text(job_dict)
        text_similarity = self.relevancy_calculator.calculate_text_similarity(candidate_text, job_text)

        candidate_domains = self.relevancy_calculator._extract_candidate_domains(candidate_dict)
        job_domain = job_dict.get('domain', 'Other')
        domain_score = self.relevancy_calculator.calculate_domain_match_score(candidate_domains, job_domain)

        overall_score = self.relevancy_calculator.calculate_comprehensive_relevancy(candidate_dict, job_dict)

        return {
            'candidate_name': candidate_dict.get('Name', 'Unknown'),
            'job_role': job_dict.get('role_title', 'Unknown Role'),
            'company': job_dict.get('company_name', ''),
            'overall_relevancy_score': overall_score,
            'skill_match_score': skill_score,
            'text_similarity_score': text_similarity,
            'domain_match_score': domain_score,
            'candidate_skills': candidate_skills,
            'job_skills': job_skills,
            'candidate_domains': candidate_domains,
            'job_domain': job_domain
        }
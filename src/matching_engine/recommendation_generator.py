"""Job recommendation generation utilities"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging

from .candidate_job_matcher import CandidateJobMatcher

logger = logging.getLogger(__name__)

class RecommendationGenerator:
    """Generate job recommendations for candidates"""

    def __init__(self):
        self.matcher = CandidateJobMatcher()

    def generate_recommendations(self, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame,
                               recommendation_type: str = "personalized", 
                               filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate job recommendations based on different strategies"""

        logger.info(f"Generating {recommendation_type} recommendations")

        # Apply filters if provided
        if filters:
            jobs_df = self._apply_filters(jobs_df, filters)

        if recommendation_type == "personalized":
            return self._generate_personalized_recommendations(candidates_df, jobs_df)
        elif recommendation_type == "domain_based":
            return self._generate_domain_based_recommendations(candidates_df, jobs_df)
        elif recommendation_type == "skill_based":
            return self._generate_skill_based_recommendations(candidates_df, jobs_df)
        else:
            raise ValueError(f"Unknown recommendation type: {recommendation_type}")

    def _generate_personalized_recommendations(self, candidates_df: pd.DataFrame, 
                                             jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate personalized recommendations using comprehensive matching"""
        return self.matcher.match_jobs_to_candidates(candidates_df, jobs_df, top_k=10)

    def _generate_domain_based_recommendations(self, candidates_df: pd.DataFrame,
                                             jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations based on domain matching"""
        recommendations = []

        for _, candidate_row in candidates_df.iterrows():
            candidate_dict = candidate_row.to_dict()
            candidate_domains = self.matcher.relevancy_calculator._extract_candidate_domains(candidate_dict)

            # Filter jobs by matching domains
            domain_matched_jobs = []
            for _, job_row in jobs_df.iterrows():
                job_dict = job_row.to_dict()
                job_domain = job_dict.get('domain', 'Other')

                if any(domain.lower() == job_domain.lower() for domain in candidate_domains):
                    domain_matched_jobs.append(job_dict)

            # If we have domain matches, use them; otherwise, fall back to all jobs
            if domain_matched_jobs:
                filtered_jobs_df = pd.DataFrame(domain_matched_jobs)
            else:
                filtered_jobs_df = jobs_df

            # Generate recommendations for this candidate
            candidate_recs = self.matcher.match_jobs_to_candidates(
                pd.DataFrame([candidate_dict]), filtered_jobs_df, top_k=5
            )

            if not candidate_recs.empty:
                recommendations.append(candidate_recs.iloc[0].to_dict())

        return pd.DataFrame(recommendations)

    def _generate_skill_based_recommendations(self, candidates_df: pd.DataFrame,
                                            jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations based primarily on skill matching"""
        recommendations = []

        for _, candidate_row in candidates_df.iterrows():
            candidate_dict = candidate_row.to_dict()
            candidate_skills = self.matcher.relevancy_calculator.skill_extractor.extract_skills_from_candidate(candidate_dict)

            if not candidate_skills:
                continue

            job_scores = []
            for _, job_row in jobs_df.iterrows():
                job_dict = job_row.to_dict()
                job_skills = self.matcher.relevancy_calculator.skill_extractor.extract_skills_from_job(job_dict)

                skill_score = self.matcher.relevancy_calculator.calculate_skill_match_score(
                    candidate_skills, job_skills
                )

                job_scores.append({
                    'JobRole': job_dict.get('role_title', 'Unknown Role'),
                    'JobId': job_dict.get('job_id', ''),
                    'Company': job_dict.get('company_name', ''),
                    'Location': job_dict.get('location', ''),
                    'Stipend': job_dict.get('stipend', ''),
                    'RelevancyScore': skill_score
                })

            # Sort by skill score and take top recommendations
            job_scores.sort(key=lambda x: x['RelevancyScore'], reverse=True)
            top_jobs = job_scores[:10]

            recommendations.append({
                'CandidateName': candidate_dict.get('Name', 'Unknown'),
                'CandidateEmail': candidate_dict.get('Email', ''),
                'RecommendedJobs': json.dumps(top_jobs)
            })

        return pd.DataFrame(recommendations)

    def _apply_filters(self, jobs_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to jobs DataFrame"""
        filtered_df = jobs_df.copy()

        if 'location' in filters and filters['location']:
            filtered_df = filtered_df[
                filtered_df['location'].str.contains(filters['location'], case=False, na=False)
            ]

        if 'job_type' in filters and filters['job_type']:
            filtered_df = filtered_df[filtered_df['job_type'] == filters['job_type']]

        if 'domain' in filters and filters['domain']:
            filtered_df = filtered_df[filtered_df['domain'] == filters['domain']]

        if 'min_stipend' in filters and filters['min_stipend']:
            # Handle stipend filtering (assuming numeric comparison)
            try:
                filtered_df = filtered_df[
                    pd.to_numeric(filtered_df['stipend'], errors='coerce') >= filters['min_stipend']
                ]
            except:
                pass

        logger.info(f"Applied filters: {len(jobs_df)} -> {len(filtered_df)} jobs")
        return filtered_df

    def generate_recommendation_summary(self, recommendations_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for recommendations"""
        if recommendations_df.empty:
            return {}

        summary = {
            'total_candidates': len(recommendations_df),
            'average_recommendations_per_candidate': 0,
            'top_companies': [],
            'top_roles': [],
            'average_relevancy_score': 0
        }

        # Calculate average recommendations per candidate
        total_recs = 0
        total_relevancy = 0
        company_counts = {}
        role_counts = {}

        for _, row in recommendations_df.iterrows():
            try:
                recommended_jobs = json.loads(row['RecommendedJobs'])
                total_recs += len(recommended_jobs)

                for job in recommended_jobs:
                    total_relevancy += job.get('RelevancyScore', 0)
                    company = job.get('Company', 'Unknown')
                    role = job.get('JobRole', 'Unknown')

                    company_counts[company] = company_counts.get(company, 0) + 1
                    role_counts[role] = role_counts.get(role, 0) + 1
            except:
                continue

        if len(recommendations_df) > 0:
            summary['average_recommendations_per_candidate'] = total_recs / len(recommendations_df)

        if total_recs > 0:
            summary['average_relevancy_score'] = total_relevancy / total_recs

        # Top companies and roles
        summary['top_companies'] = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        summary['top_roles'] = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return summary
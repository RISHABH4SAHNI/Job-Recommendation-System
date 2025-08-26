"""Job recommendation API routes"""
from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import json

from ..models.schemas import (
    RecommendationRequest, 
    JobRecommendation, 
    CandidateProfile,
    RelevancyScore
)
from ...matching_engine import RecommendationGenerator
from config.settings import RAW_DATA_DIR
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global instances
recommendation_generator = None
jobs_df = None

def get_recommendation_generator():
    """Get or initialize the recommendation generator"""
    global recommendation_generator, jobs_df

    if recommendation_generator is None:
        recommendation_generator = RecommendationGenerator()

        # Load jobs data
        try:
            jobs_df = pd.read_csv(RAW_DATA_DIR / "job_details.csv")
            logger.info(f"Loaded {len(jobs_df)} jobs")
        except Exception as e:
            logger.error(f"Error loading jobs data: {e}")
            raise HTTPException(
                status_code=503,
                detail="Jobs data not available"
            )

    return recommendation_generator, jobs_df

@router.post("/generate", response_model=List[JobRecommendation])
async def generate_recommendations(request: RecommendationRequest):
    """Generate job recommendations for a candidate"""
    try:
        generator, jobs_data = get_recommendation_generator()

        # Convert candidate to DataFrame format
        candidate_dict = {
            'Name': request.candidate.name,
            'Email': request.candidate.email,
            'HardSkills': json.dumps([{'skill': skill, 'percentage': 80} for skill in request.candidate.skills]),
            'Experiences': request.candidate.experiences or '',
            'Major': request.candidate.education or '',
            'RecommendedJobDomains': json.dumps([{'job_domain': domain} for domain in request.candidate.domains])
        }

        candidate_df = pd.DataFrame([candidate_dict])

        # Generate recommendations
        recommendations_df = generator.generate_recommendations(
            candidate_df, 
            jobs_data,
            recommendation_type="personalized",
            filters=request.filters
        )

        if recommendations_df.empty:
            return []

        # Parse the recommendations JSON
        recommended_jobs_str = recommendations_df.iloc[0]['RecommendedJobs']
        recommended_jobs = json.loads(recommended_jobs_str)

        # Convert to response format
        recommendations = []
        for job in recommended_jobs[:request.max_recommendations]:
            recommendations.append(JobRecommendation(
                job_id=job.get('JobId', ''),
                title=job.get('JobRole', ''),
                company=job.get('Company', ''),
                location=job.get('Location', ''),
                relevancy_score=job.get('RelevancyScore', 0.0)
            ))

        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Job Recommendation API"}
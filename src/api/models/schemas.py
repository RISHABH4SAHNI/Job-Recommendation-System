"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any

class JobDescription(BaseModel):
    """Job description input model"""
    description: str

class JobDomainPrediction(BaseModel):
    """Job domain prediction response"""
    domain: str
    confidence: float
    cleaned_text: str

class CandidateProfile(BaseModel):
    """Candidate profile model"""
    name: str
    email: Optional[EmailStr] = None
    skills: List[str] = []
    experiences: Optional[str] = None
    education: Optional[str] = None
    domains: List[str] = []

class JobPosting(BaseModel):
    """Job posting model"""
    job_id: str
    title: str
    company: str
    description: str
    requirements: Optional[str] = None
    location: Optional[str] = None
    stipend: Optional[str] = None
    domain: Optional[str] = None

class RelevancyScore(BaseModel):
    """Relevancy score model"""
    candidate_name: str
    job_title: str
    overall_score: float
    skill_score: float
    text_similarity: float
    domain_score: float

class JobRecommendation(BaseModel):
    """Job recommendation model"""
    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    relevancy_score: float

class RecommendationRequest(BaseModel):
    """Job recommendation request"""
    candidate: CandidateProfile
    max_recommendations: int = 10
    filters: Optional[Dict[str, Any]] = None
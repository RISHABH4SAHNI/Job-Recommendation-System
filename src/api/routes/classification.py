"""Job classification API routes"""
from fastapi import APIRouter, HTTPException
from ..models.schemas import JobDescription, JobDomainPrediction
from ...job_domain_classifier import JobDomainClassifier
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classification", tags=["classification"])

# Global classifier instance
classifier = None

def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        classifier = JobDomainClassifier()
        try:
            classifier.load_model()
            logger.info("Loaded pre-trained model")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first."
            )
    return classifier

@router.post("/predict", response_model=JobDomainPrediction)
async def predict_job_domain(job: JobDescription):
    """Predict job domain from job description"""
    try:
        classifier = get_classifier()

        # Clean the text for response
        cleaned_text = classifier.preprocessor.clean_text(job.description)

        # Make prediction
        predicted_domain = classifier.predict(job.description)

        logger.info(f"Predicted domain: {predicted_domain}")

        return JobDomainPrediction(
            domain=predicted_domain,
            confidence=0.8,  # You could get actual confidence from the model
            cleaned_text=cleaned_text
        )
    except Exception as e:
        logger.error(f"Error in domain prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""FastAPI main application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import classification, recommendations
from config.settings import API_HOST, API_PORT
from config.logging_config import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Job Recommendation System API",
    description="API for job domain classification and candidate-job matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classification.router)
app.include_router(recommendations.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Job Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
#!/usr/bin/env python3
"""Script to train job domain classification models"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from job_domain_classifier import JobDomainClassifier
from data_processing import CSVProcessor
from config.settings import RAW_DATA_DIR, MODELS_DIR
from config.logging_config import setup_logging

def main():
    """Main training function"""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting model training process")

    try:
        # Load and process data
        processor = CSVProcessor()
        jobs_df = processor.load_job_details()

        if jobs_df.empty:
            logger.error("No job data found. Please ensure job_details.csv exists in data/raw/")
            return

        logger.info(f"Loaded {len(jobs_df)} job records")

        # Initialize classifier
        classifier = JobDomainClassifier()

        # Prepare data
        X, y = classifier.prepare_data(jobs_df, text_column='role_description')

        if len(X) == 0:
            logger.error("No data available for training after preprocessing")
            return

        # Train models
        logger.info("Training models...")
        results = classifier.train_models(X, y)

        # Display results
        logger.info("Training Results:")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}")

        # Save best model
        classifier.save_model(MODELS_DIR)
        logger.info(f"Best model ({classifier.model_name}) saved to {MODELS_DIR}")

        # Save processed data
        processed_jobs = classifier.preprocessor.preprocess_job_data(jobs_df)
        processor.save_processed_data(processed_jobs, "processed_jobs.csv")

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
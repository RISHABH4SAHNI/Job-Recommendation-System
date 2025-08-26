#!/usr/bin/env python3
"""Script to process and organize data files"""

import sys
from pathlib import Path
import pandas as pd
import shutil
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing import CSVProcessor
from matching_engine import CandidateJobMatcher
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, BASE_DIR
from config.logging_config import setup_logging

def organize_existing_data():
    """Move existing CSV files to raw data directory"""
    logger = logging.getLogger(__name__)

    csv_files = [
        "job_details.csv",
        "all_resumes_data.csv",
        "job_details_with_predictions.csv",
        "job_relevancy_scores_with_candidates.csv",
        "detailed_job_relevancy_scores.json.csv",
        "matched_jobs_candidates.csv"
    ]

    for csv_file in csv_files:
        source_path = BASE_DIR / csv_file
        if source_path.exists():
            dest_path = RAW_DATA_DIR / csv_file
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Moved {csv_file} to {RAW_DATA_DIR}")

def process_job_data():
    """Process job data"""
    logger = logging.getLogger(__name__)
    processor = CSVProcessor()

    # Load and clean job data
    jobs_df = processor.load_job_details()

    if not jobs_df.empty:
        # Merge with predictions if available
        jobs_df = processor.merge_job_predictions(jobs_df)

        # Save processed data
        processor.save_processed_data(jobs_df, "processed_jobs.csv")

        # Print summary
        summary = processor.get_data_summary(jobs_df)
        logger.info(f"Job data summary: {summary['total_records']} records")
        logger.info(f"Domain distribution: {summary.get('domain_distribution', {})}")

    return jobs_df

def process_candidate_data():
    """Process candidate data"""
    logger = logging.getLogger(__name__)
    processor = CSVProcessor()

    # Load and clean candidate data
    candidates_df = processor.load_candidate_data()

    if not candidates_df.empty:
        # Save processed data
        processor.save_processed_data(candidates_df, "processed_candidates.csv")

        # Print summary
        summary = processor.get_data_summary(candidates_df)
        logger.info(f"Candidate data summary: {summary['total_records']} records")

    return candidates_df

def main():
    """Main data processing function"""
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting data processing")

    # Organize existing data
    organize_existing_data()

    # Process job and candidate data
    jobs_df = process_job_data()
    candidates_df = process_candidate_data()

    logger.info("Data processing completed!")

if __name__ == "__main__":
    main()
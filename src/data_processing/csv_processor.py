"""CSV data processing utilities"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Process and clean CSV data files"""

    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR

    def load_job_details(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and clean job details CSV"""
        if file_path is None:
            file_path = self.raw_data_dir / "job_details.csv"

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} job records from {file_path}")

            # Basic cleaning
            df = self._clean_job_data(df)

            return df
        except Exception as e:
            logger.error(f"Error loading job details: {e}")
            return pd.DataFrame()

    def load_candidate_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and clean candidate data CSV"""
        if file_path is None:
            file_path = self.raw_data_dir / "all_resumes_data.csv"

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} candidate records from {file_path}")

            # Basic cleaning
            df = self._clean_candidate_data(df)

            return df
        except Exception as e:
            logger.error(f"Error loading candidate data: {e}")
            return pd.DataFrame()

    def _clean_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean job data DataFrame"""
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['job_id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate job records")

        # Handle missing values
        df['role_description'] = df['role_description'].fillna('')
        df['requirement'] = df['requirement'].fillna('')
        df['description'] = df['description'].fillna('')
        df['location'] = df['location'].fillna('Remote')
        df['company_name'] = df['company_name'].fillna('Unknown')

        # Clean stipend column
        df['stipend'] = df['stipend'].astype(str).str.replace(',', '')

        # Create combined description
        df['combined_description'] = (
            df['role_description'].astype(str) + ' ' + 
            df['requirement'].astype(str) + ' ' + 
            df['description'].astype(str)
        )

        return df

    def _clean_candidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean candidate data DataFrame"""
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Email'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate candidate records")

        # Handle missing values
        df['Name'] = df['Name'].fillna('Unknown')
        df['Email'] = df['Email'].fillna('')
        df['Experiences'] = df['Experiences'].fillna('[]')
        df['Projects'] = df['Projects'].fillna('[]')
        df['HardSkills'] = df['HardSkills'].fillna('[]')
        df['SoftSkills'] = df['SoftSkills'].fillna('[]')
        df['RecommendedJobDomains'] = df['RecommendedJobDomains'].fillna('[]')

        return df

    def merge_job_predictions(self, jobs_df: pd.DataFrame, predictions_file: Optional[Path] = None) -> pd.DataFrame:
        """Merge job data with domain predictions"""
        if predictions_file is None:
            predictions_file = self.raw_data_dir / "job_details_with_predictions.csv"

        try:
            predictions_df = pd.read_csv(predictions_file)

            # Merge on job_id
            merged_df = jobs_df.merge(
                predictions_df[['job_id', 'predicted_domain']], 
                on='job_id', 
                how='left'
            )

            # Use predicted_domain as domain if available
            merged_df['domain'] = merged_df['predicted_domain'].fillna('Other')

            logger.info(f"Merged job data with predictions: {len(merged_df)} records")
            return merged_df

        except Exception as e:
            logger.warning(f"Could not merge predictions: {e}")
            jobs_df['domain'] = 'Other'
            return jobs_df

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        output_path = self.processed_data_dir / filename

        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of DataFrame"""
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }

        # Add specific summaries based on data type
        if 'domain' in df.columns:
            summary['domain_distribution'] = df['domain'].value_counts().to_dict()

        if 'company_name' in df.columns:
            summary['top_companies'] = df['company_name'].value_counts().head(10).to_dict()

        return summary
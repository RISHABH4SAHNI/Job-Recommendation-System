"""File handling utilities"""
import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file operations"""

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Path):
        """Save data to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved JSON data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {e}")

    @staticmethod
    def load_json(file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}

    @staticmethod
    def save_pickle(data: Any, file_path: Path):
        """Save data using pickle"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved pickle data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving pickle file {file_path}: {e}")

    @staticmethod
    def load_pickle(file_path: Path) -> Any:
        """Load data from pickle file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded pickle data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading pickle file {file_path}: {e}")
            return None

    @staticmethod
    def save_model(model: Any, file_path: Path):
        """Save ML model using joblib"""
        try:
            joblib.dump(model, file_path)
            logger.info(f"Saved model to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model {file_path}: {e}")

    @staticmethod
    def load_model(file_path: Path) -> Any:
        """Load ML model using joblib"""
        try:
            model = joblib.load(file_path)
            logger.info(f"Loaded model from {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {file_path}: {e}")
            return None

    @staticmethod
    def ensure_directory(directory_path: Path):
        """Ensure directory exists"""
        directory_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_file_size(file_path: Path) -> str:
        """Get human readable file size"""
        try:
            size_bytes = file_path.stat().st_size

            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        except Exception:
            return "Unknown"

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
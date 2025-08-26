"""Job Domain Classifier using machine learning models"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from config.settings import MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, MAX_FEATURES, NGRAM_RANGE
from .preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

class JobDomainClassifier:
    """Job Domain Classifier with multiple ML models"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        self.models = {
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'svm': SVC(random_state=RANDOM_STATE)
        }
        self.best_model = None
        self.model_name = None

    def prepare_data(self, df: pd.DataFrame, text_column: str = "role_description") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for training")

        # Preprocess data
        processed_df = self.preprocessor.preprocess_job_data(df, text_column)

        # Extract features and target
        X = processed_df["cleaned_text"]
        y = processed_df["domain"]

        # Vectorize text
        X_vec = self.vectorizer.fit_transform(X)

        # Handle class imbalance
        oversampler = RandomOverSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = oversampler.fit_resample(X_vec, y)

        logger.info(f"Data prepared: {X_resampled.shape[0]} samples, {X_resampled.shape[1]} features")
        return X_resampled, y_resampled

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and return performance metrics"""
        logger.info("Training models")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        results = {}

        # Train Random Forest with hyperparameter tuning
        logger.info("Training Random Forest with hyperparameter tuning")
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        grid_search_rf = GridSearchCV(
            self.models['random_forest'], 
            param_grid_rf, 
            cv=CV_FOLDS, 
            n_jobs=-1, 
            verbose=1
        )
        grid_search_rf.fit(X_train, y_train)

        # Use best Random Forest model
        self.models['random_forest'] = grid_search_rf.best_estimator_

        # Train other models
        for name, model in self.models.items():
            if name != 'random_forest':  # RF already trained above
                logger.info(f"Training {name}")
                model.fit(X_train, y_train)

            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report
            }

            logger.info(f"{name} accuracy: {accuracy:.4f}")

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.model_name = best_model_name

        logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        return results

    def predict(self, text: str) -> str:
        """Predict job domain for given text"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models first.")

        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)

        # Vectorize
        X_vec = self.vectorizer.transform([cleaned_text])

        # Predict
        prediction = self.best_model.predict(X_vec)[0]
        return prediction

    def save_model(self, model_path: Path = None):
        """Save trained model and vectorizer"""
        if model_path is None:
            model_path = MODELS_DIR

        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model and vectorizer
        joblib.dump(self.best_model, model_path / f"best_{self.model_name}_classifier.joblib")
        joblib.dump(self.vectorizer, model_path / "tfidf_vectorizer.joblib")

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: Path = None):
        """Load trained model and vectorizer"""
        if model_path is None:
            model_path = MODELS_DIR

        model_path = Path(model_path)

        # Try to load Random Forest first (likely the best performing)
        model_files = list(model_path.glob("*random_forest*.joblib"))
        if model_files:
            self.best_model = joblib.load(model_files[0])
            self.model_name = "random_forest"
        else:
            # Load any available model
            model_files = list(model_path.glob("*classifier*.joblib"))
            if model_files:
                self.best_model = joblib.load(model_files[0])
                self.model_name = model_files[0].stem.replace("best_", "").replace("_classifier", "")

        # Load vectorizer
        vectorizer_path = model_path / "tfidf_vectorizer.joblib"
        if vectorizer_path.exists():
            self.vectorizer = joblib.load(vectorizer_path)

        logger.info(f"Model loaded from {model_path}")
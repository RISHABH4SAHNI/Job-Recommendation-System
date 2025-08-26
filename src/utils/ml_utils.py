"""Machine learning utilities"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MLUtils:
    """Utility class for machine learning operations"""

    @staticmethod
    def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray, 
                                    labels: List[str] = None) -> Dict[str, Any]:
        """Evaluate classification model performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics['confusion_matrix'] = cm.tolist()

        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str] = None, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    @staticmethod
    def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation on model"""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"Cross-validation completed. Mean {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        return results

    @staticmethod
    def feature_importance_analysis(model, feature_names: List[str], 
                                  top_k: int = 20) -> pd.DataFrame:
        """Analyze feature importance for tree-based models"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                feature_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                logger.info(f"Feature importance analysis completed for {len(feature_names)} features")
                return feature_df.head(top_k)
            else:
                logger.warning("Model doesn't have feature_importances_ attribute")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            return pd.DataFrame()

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, save_path: str = None):
        """Plot feature importance"""
        if importance_df.empty:
            logger.warning("No feature importance data to plot")
            return

        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    @staticmethod
    def split_data_by_time(df: pd.DataFrame, date_column: str, 
                          train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically"""
        df_sorted = df.sort_values(date_column)
        split_index = int(len(df_sorted) * train_ratio)

        train_df = df_sorted.iloc[:split_index]
        test_df = df_sorted.iloc[split_index:]

        logger.info(f"Time-based split: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df

    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[Any, float]:
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        class_weight_dict = dict(zip(classes, weights))
        logger.info(f"Calculated class weights: {class_weight_dict}")

        return class_weight_dict

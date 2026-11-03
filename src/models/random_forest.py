"""
Random Forest Regression Module for Transformer Telemetry System.
Implements modular model training, prediction, evaluation, and persistence
using scikit-learn's RandomForestRegressor.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> Dict[str, float]:
    """
    Computes regression evaluation metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination
    - MAPE_epsilon: Mean Absolute Percentage Error with epsilon floor to avoid div-by-zero
    - wMAPE: Weighted MAPE (sum(|y - y_hat|) / sum(y) * 100), standard in power grid forecasting
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Avoid div-by-zero on zero actual readings
    denominator = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)

    # Weighted MAPE: total absolute error divided by total actual demand
    sum_true = float(np.sum(np.abs(y_true)))
    wmape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0) if sum_true > 0 else 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'wmape': wmape
    }


class TransformerRandomForest:
    """
    Wrapper around RandomForestRegressor for transformer power and load percentage forecasting.
    """

    def __init__(
        self,
        transformer_id: str,
        target_name: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.transformer_id = transformer_id.upper()
        self.target_name = target_name
        self.random_state = random_state
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        self.model = RandomForestRegressor(**self.params)
        self.feature_names: List[str] = []
        self.is_trained: bool = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the Random Forest model on training features and target."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for feature matrix X."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")
        return self.model.predict(X[self.feature_names])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluates model on test features and target."""
        preds = self.predict(X_test)
        metrics = calculate_metrics(y_test.values, preds)
        return metrics

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importances as a pandas DataFrame."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before extracting feature importances.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return df

    def save(self, filepath: str) -> None:
        """Persists trained model and metadata using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'transformer_id': self.transformer_id,
            'target_name': self.target_name,
            'params': self.params,
            'feature_names': self.feature_names,
            'model': self.model
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TransformerRandomForest':
        """Loads a persisted model bundle from disk."""
        bundle = joblib.load(filepath)
        instance = cls(
            transformer_id=bundle['transformer_id'],
            target_name=bundle['target_name'],
            **bundle['params']
        )
        instance.model = bundle['model']
        instance.feature_names = bundle['feature_names']
        instance.is_trained = True
        return instance


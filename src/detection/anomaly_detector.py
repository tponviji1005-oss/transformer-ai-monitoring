"""
Anomaly-Based Overload Warning Module for Transformer T1.
Implements unsupervised anomaly detection using Isolation Forest trained strictly
on normal operating data (2013–2016, zero overloads) to flag out-of-distribution
operating conditions and impending load surges in 2017.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class TransformerAnomalyDetector:
    """
    Unsupervised Isolation Forest detector for identifying anomalous transformer behavior.
    Trained strictly on normal historical training data (where no overloads occurred).
    """

    def __init__(
        self,
        transformer_id: str = "T1",
        n_estimators: int = 150,
        contamination: float = 0.01,
        max_samples: Union[str, float, int] = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.transformer_id = transformer_id.upper()
        self.contamination = contamination
        self.random_state = random_state
        self.params = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'random_state': random_state,
            'n_jobs': n_jobs,
        }
        self.feature_names: List[str] = []
        self.model: Optional[IsolationForest] = None
        self.train_score_min_: float = 0.0
        self.train_score_max_: float = 1.0
        self.is_trained: bool = False

    def train(self, X_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits the Isolation Forest on normal historical operating data.
        """
        self.feature_names = list(X_train.columns)
        self.model = IsolationForest(**self.params)
        self.model.fit(X_train[self.feature_names])

        # Record baseline decision function statistics on training data
        train_raw_scores = self.model.decision_function(X_train[self.feature_names])
        self.train_score_min_ = float(np.min(train_raw_scores))
        self.train_score_max_ = float(np.max(train_raw_scores))
        self.is_trained = True

        return {
            'transformer_id': self.transformer_id,
            'train_samples': len(X_train),
            'contamination': self.contamination,
            'n_estimators': self.params['n_estimators'],
            'train_score_min': self.train_score_min_,
            'train_score_max': self.train_score_max_,
        }

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns raw decision function scores.
        Negative values represent anomalies; positive values represent inliers.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Anomaly detector must be trained before scoring.")
        return self.model.decision_function(X[self.feature_names])

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a normalized anomaly risk index in [0, 1], where 1.0 indicates maximum anomaly.
        Derived by inverting and scaling the decision function score.
        """
        raw = self.decision_function(X)
        # Decision function typically spans [-0.4, 0.4]. Anomaly score = 0.5 - raw
        # Sigmoid or linear clamping:
        # Standard approach: map raw < 0 to > 0.5 risk
        clipped = np.clip(raw, -0.5, 0.5)
        # Risk score: 0.5 - raw (so negative raw becomes > 0.5)
        risk = 0.5 - clipped
        return np.clip(risk, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns binary anomaly predictions: 1 for anomaly, 0 for normal inlier.
        (Converts IsolationForest -1/1 output to 1/0).
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Anomaly detector must be trained before predicting.")
        preds = self.model.predict(X[self.feature_names])
        return (preds == -1).astype(int)

    def evaluate_against_overloads(
        self,
        X_test: pd.DataFrame,
        y_overload: pd.Series,
        overload_type: str = "current_overload",
    ) -> Dict[str, Any]:
        """
        Evaluates how effectively detected anomalies align with actual overload events.
        Clearly reports precision, recall, and false positive characteristics.
        """
        y_true = np.asarray(y_overload, dtype=int)
        anomalies = self.predict(X_test)
        scores = self.anomaly_score(X_test)

        total_samples = len(y_true)
        total_overloads = int(y_true.sum())
        total_anomalies = int(anomalies.sum())

        overloads_caught = int(np.sum((anomalies == 1) & (y_true == 1)))
        overload_recall = float(overloads_caught / total_overloads) if total_overloads > 0 else 0.0
        precision_on_overloads = float(overloads_caught / total_anomalies) if total_anomalies > 0 else 0.0
        f1 = float(
            2 * precision_on_overloads * overload_recall / (precision_on_overloads + overload_recall + 1e-9)
        )

        cm = confusion_matrix(y_true, anomalies)

        # Score distributions
        mean_score_overload = float(np.mean(scores[y_true == 1])) if total_overloads > 0 else 0.0
        mean_score_normal = float(np.mean(scores[y_true == 0])) if (total_samples - total_overloads) > 0 else 0.0

        return {
            'overload_type': overload_type,
            'total_samples': total_samples,
            'total_anomalies_flagged': total_anomalies,
            'anomaly_rate_pct': float(total_anomalies / total_samples * 100.0),
            'actual_overloads': total_overloads,
            'overloads_detected_as_anomalies': overloads_caught,
            'overload_detection_recall': overload_recall,
            'precision_on_overloads': precision_on_overloads,
            'f1_score': f1,
            'confusion_matrix': {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]), # Anomalies that are NOT overloads (e.g. other anomalies)
                'fn': int(cm[1, 0]), # Overloads missed by anomaly detector
                'tp': int(cm[1, 1]), # Overloads caught as anomalies
            },
            'mean_anomaly_score_overload': mean_score_overload,
            'mean_anomaly_score_normal': mean_score_normal,
        }

    def save(self, filepath: str) -> None:
        """Persists trained anomaly detector to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'transformer_id': self.transformer_id,
            'params': self.params,
            'feature_names': self.feature_names,
            'train_score_min': self.train_score_min_,
            'train_score_max': self.train_score_max_,
            'model': self.model,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TransformerAnomalyDetector':
        """Loads a persisted anomaly detector from disk."""
        bundle = joblib.load(filepath)
        instance = cls(
            transformer_id=bundle['transformer_id'],
            **bundle['params']
        )
        instance.model = bundle['model']
        instance.feature_names = bundle['feature_names']
        instance.train_score_min_ = bundle.get('train_score_min', 0.0)
        instance.train_score_max_ = bundle.get('train_score_max', 1.0)
        instance.is_trained = True
        return instance


"""
Supervised Overload Classification Module for Transformer T2.
Implements Random Forest Classification for predicting next-hour transformer overload
with training-only threshold tuning to prioritize recall for dangerous overload events.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit


class TransformerOverloadClassifier:
    """
    Random Forest Classifier for detecting impending transformer overloads.
    Trained strictly on historical training data with class-weight balancing
    and cross-validated threshold selection prioritizing recall.
    """

    def __init__(
        self,
        transformer_id: str = "T2",
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        default_threshold: float = 0.5,
    ):
        self.transformer_id = transformer_id.upper()
        self.random_state = random_state
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
        }
        self.default_threshold = default_threshold
        self.selected_threshold = default_threshold
        self.feature_names: List[str] = []
        self.model: Optional[RandomForestClassifier] = None
        self.cv_metrics_: Dict[str, float] = {}
        self.is_trained: bool = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_threshold: bool = True,
        cv_folds: int = 5,
        target_recall: float = 0.85,
        cv_strategy: str = "stratified",
    ) -> Dict[str, Any]:
        """
        Trains the classifier strictly using training data.
        Performs Cross-Validation on training data to evaluate out-of-fold probability
        and select a decision threshold that achieves target recall while maximizing precision.
        
        cv_strategy options:
        - 'stratified': Shuffled StratifiedKFold preserving rare overload class distribution (default for baseline 0.3382 threshold).
        - 'chronological': TimeSeriesSplit preventing mixing of earlier and later time periods.
        """
        self.feature_names = list(X_train.columns)
        y_train_arr = np.asarray(y_train, dtype=int)

        if tune_threshold:
            if cv_strategy.lower() == "chronological":
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
                oof_probs = []
                val_indices = []
                for train_idx, val_idx in cv_splitter.split(X_train):
                    fold_clf = RandomForestClassifier(**self.params)
                    fold_clf.fit(X_train.iloc[train_idx], y_train_arr[train_idx])
                    p = fold_clf.predict_proba(X_train.iloc[val_idx])[:, 1]
                    oof_probs.extend(p)
                    val_indices.extend(val_idx)
                oof_probs = np.array(oof_probs, dtype=float)
                y_eval_arr = y_train_arr[val_indices]
            else:
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                oof_probs = np.zeros(len(X_train), dtype=float)
                for train_idx, val_idx in cv_splitter.split(X_train, y_train_arr):
                    fold_clf = RandomForestClassifier(**self.params)
                    fold_clf.fit(X_train.iloc[train_idx], y_train_arr[train_idx])
                    oof_probs[val_idx] = fold_clf.predict_proba(X_train.iloc[val_idx])[:, 1]
                y_eval_arr = y_train_arr

            # Cross-validated training metrics
            oof_roc = float(roc_auc_score(y_eval_arr, oof_probs))
            oof_pr = float(average_precision_score(y_eval_arr, oof_probs))

            # Threshold tuning targeting recall on training data only
            precisions, recalls, thresholds = precision_recall_curve(y_eval_arr, oof_probs)
            
            # Find candidate thresholds where recall >= target_recall
            valid_mask = recalls[:-1] >= target_recall
            if np.any(valid_mask):
                # Pick the threshold among valid candidates that gives highest precision
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(precisions[valid_indices])]
                self.selected_threshold = float(thresholds[best_idx])
            else:
                # Fallback to F2 score (weights recall twice as much as precision)
                f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
                best_f2_idx = int(np.argmax(f2_scores[:-1]))
                self.selected_threshold = float(thresholds[best_f2_idx])

            # Evaluate OOF at default (0.5) and selected threshold
            pred_default = (oof_probs >= self.default_threshold).astype(int)
            pred_selected = (oof_probs >= self.selected_threshold).astype(int)

            self.cv_metrics_ = {
                'cv_strategy': cv_strategy,
                'oof_roc_auc': oof_roc,
                'oof_pr_auc': oof_pr,
                'selected_threshold': self.selected_threshold,
                'default_threshold': self.default_threshold,
                'oof_recall_default': float(recall_score(y_eval_arr, pred_default)),
                'oof_precision_default': float(precision_score(y_eval_arr, pred_default, zero_division=0)),
                'oof_f1_default': float(f1_score(y_eval_arr, pred_default, zero_division=0)),
                'oof_recall_selected': float(recall_score(y_eval_arr, pred_selected)),
                'oof_precision_selected': float(precision_score(y_eval_arr, pred_selected, zero_division=0)),
                'oof_f1_selected': float(f1_score(y_eval_arr, pred_selected, zero_division=0)),
            }
        else:
            self.selected_threshold = self.default_threshold

        # Fit final model on full training set
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train[self.feature_names], y_train_arr)
        self.is_trained = True

        return {
            'transformer_id': self.transformer_id,
            'train_samples': len(X_train),
            'train_overloads': int(y_train_arr.sum()),
            'overload_prevalence_pct': float(y_train_arr.mean() * 100.0),
            'selected_threshold': self.selected_threshold,
            'cv_metrics': self.cv_metrics_,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted probability of overload [0, 1]."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict_proba.")
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary overload predictions using specified or selected threshold."""
        t = self.selected_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the classifier on a test dataset.
        Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
        """
        y_test_arr = np.asarray(y_test, dtype=int)
        probs = self.predict_proba(X_test)
        t = self.selected_threshold if threshold is None else threshold
        preds = (probs >= t).astype(int)

        roc_auc = float(roc_auc_score(y_test_arr, probs))
        pr_auc = float(average_precision_score(y_test_arr, probs))
        acc = float(accuracy_score(y_test_arr, preds))
        prec = float(precision_score(y_test_arr, preds, zero_division=0))
        rec = float(recall_score(y_test_arr, preds, zero_division=0))
        f1 = float(f1_score(y_test_arr, preds, zero_division=0))
        cm = confusion_matrix(y_test_arr, preds)

        return {
            'threshold_used': float(t),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1]),
            },
            'test_samples': len(y_test_arr),
            'actual_overloads': int(y_test_arr.sum()),
            'predicted_overloads': int(preds.sum()),
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importances."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before extracting feature importances.")
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Persists trained classifier and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'transformer_id': self.transformer_id,
            'params': self.params,
            'feature_names': self.feature_names,
            'default_threshold': self.default_threshold,
            'selected_threshold': self.selected_threshold,
            'cv_metrics': self.cv_metrics_,
            'model': self.model,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TransformerOverloadClassifier':
        """Loads a persisted classifier from disk."""
        bundle = joblib.load(filepath)
        instance = cls(
            transformer_id=bundle['transformer_id'],
            default_threshold=bundle.get('default_threshold', 0.5),
            **bundle['params']
        )
        instance.model = bundle['model']
        instance.feature_names = bundle['feature_names']
        instance.selected_threshold = bundle['selected_threshold']
        instance.cv_metrics_ = bundle.get('cv_metrics', {})
        instance.is_trained = True
        return instance


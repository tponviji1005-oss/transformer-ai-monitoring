"""
Unit and Integration Tests for Milestone 4 Overload Detection System.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.overload_classifier import TransformerOverloadClassifier
from src.detection.anomaly_detector import TransformerAnomalyDetector
from src.detection.threshold_monitor import ThresholdMonitor, MonitoringStatus
from src.detection.alert_engine import UnifiedAlertEngine, AlertLevel, AlertType, TransformerAlert


class TestOverloadDetectionSystem(unittest.TestCase):

    def test_t2_classifier_load_and_predict(self):
        model_path = os.path.join('models', 'overload', 't2_overload_classifier.joblib')
        self.assertTrue(os.path.exists(model_path), "T2 classifier artifact must exist")
        clf = TransformerOverloadClassifier.load(model_path)
        self.assertTrue(clf.is_trained)
        self.assertEqual(len(clf.feature_names), 39)
        self.assertGreater(clf.selected_threshold, 0.0)
        self.assertLess(clf.selected_threshold, 1.0)

        # Mock sample
        mock_data = pd.DataFrame(np.zeros((2, 39)), columns=clf.feature_names)
        probs = clf.predict_proba(mock_data)
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        preds = clf.predict(mock_data)
        self.assertEqual(len(preds), 2)

    def test_t1_anomaly_detector_load_and_predict(self):
        model_path = os.path.join('models', 'overload', 't1_anomaly_detector.joblib')
        self.assertTrue(os.path.exists(model_path), "T1 anomaly detector artifact must exist")
        det = TransformerAnomalyDetector.load(model_path)
        self.assertTrue(det.is_trained)
        self.assertEqual(len(det.feature_names), 39)

        # Mock sample
        mock_data = pd.DataFrame(np.zeros((2, 39)), columns=det.feature_names)
        preds = det.predict(mock_data)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p in (0, 1) for p in preds))
        scores = det.anomaly_score(mock_data)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_t3_threshold_monitor(self):
        monitor = ThresholdMonitor(transformer_id="T3", warning_load_pct=80.0, critical_load_pct=100.0)
        
        # Test Normal
        res_normal = monitor.evaluate_sample(load_pct=65.0, oil_temp_C=50.0)
        self.assertEqual(res_normal['status'], MonitoringStatus.NORMAL.value)
        
        # Test Warning
        res_warning = monitor.evaluate_sample(load_pct=85.0, oil_temp_C=60.0)
        self.assertEqual(res_warning['status'], MonitoringStatus.WARNING.value)
        
        # Test Critical
        res_crit = monitor.evaluate_sample(load_pct=105.0, oil_temp_C=70.0)
        self.assertEqual(res_crit['status'], MonitoringStatus.CRITICAL.value)

    def test_unified_alert_engine_priority(self):
        engine = UnifiedAlertEngine(warning_load_pct=80.0, critical_load_pct=100.0, classifier_alert_threshold=0.35)

        # 1. Measured Overload -> CRITICAL
        a1 = engine.generate_alert("T2", "2017-06-01 12:00:00", current_load_pct=108.5, predicted_load_pct=95.0)
        self.assertEqual(a1.risk_level, AlertLevel.CRITICAL.value)
        self.assertEqual(a1.alert_type, AlertType.CURRENT_OVERLOAD.value)
        self.assertEqual(
            a1.recommended_action,
            "Notify the substation operator for advisory review. Recommend manual feeder inspection and assessment of load redistribution options, following approved operating procedures."
        )
        # Ensure purely advisory phrasing without automatic tripping or shedding commands
        self.assertNotIn("initiate load shedding", a1.recommended_action)
        self.assertNotIn("automatic", a1.recommended_action.lower())

        # 2. Predicted Overload via high classifier prob -> WARNING / CRITICAL
        a2 = engine.generate_alert("T2", "2017-06-01 13:00:00", current_load_pct=75.0, overload_probability=0.55)
        self.assertEqual(a2.alert_type, AlertType.PREDICTED_OVERLOAD.value)

        # 3. Elevated Load -> WARNING
        a3 = engine.generate_alert("T3", "2017-06-01 14:00:00", current_load_pct=88.0)
        self.assertEqual(a3.risk_level, AlertLevel.WARNING.value)
        self.assertEqual(a3.alert_type, AlertType.ELEVATED_LOAD_WARNING.value)

        # 4. Anomaly detected -> WARNING
        a4 = engine.generate_alert("T1", "2017-06-01 15:00:00", current_load_pct=60.0, is_anomaly=True, anomaly_score=0.65)
        self.assertEqual(a4.risk_level, AlertLevel.WARNING.value)
        self.assertEqual(a4.alert_type, AlertType.LOAD_ANOMALY.value)

        # 5. Normal -> NORMAL
        a5 = engine.generate_alert("T3", "2017-06-01 16:00:00", current_load_pct=50.0)
        self.assertEqual(a5.risk_level, AlertLevel.NORMAL.value)
        self.assertEqual(a5.alert_type, AlertType.NORMAL_OPERATION.value)


if __name__ == '__main__':
    unittest.main()


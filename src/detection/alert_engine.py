"""
Unified Alert Engine for Transformer Monitoring and Early Warning.
Combines measured telemetry, classification predictions, anomaly signals, and
regression forecasts across T1, T2, and T3 to generate standardized, actionable alerts.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    CURRENT_OVERLOAD = "CURRENT_OVERLOAD"
    PREDICTED_OVERLOAD = "PREDICTED_OVERLOAD"
    ELEVATED_LOAD_WARNING = "ELEVATED_LOAD_WARNING"
    LOAD_ANOMALY = "LOAD_ANOMALY"
    NORMAL_OPERATION = "NORMAL_OPERATION"


@dataclass
class TransformerAlert:
    """Standardized Alert Representation."""
    transformer_id: str
    timestamp: str
    current_load_pct: float
    predicted_load_pct: Optional[float]
    risk_level: str
    alert_type: str
    reason: str
    recommended_action: str
    overload_probability: Optional[float] = None
    anomaly_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts alert to a standard dictionary."""
        return asdict(self)

    def to_formatted_message(self) -> str:
        """Renders alert as human-readable multi-line text for console/operator view."""
        pred_str = f"{self.predicted_load_pct:.1f}%" if self.predicted_load_pct is not None else "N/A"
        lines = [
            f"=== TRANSFORMER ALERT [{self.risk_level}] ===",
            f"Transformer:        {self.transformer_id}",
            f"Timestamp:          {self.timestamp}",
            f"Current Load:       {self.current_load_pct:.1f}%",
            f"Predicted Next Load:{pred_str}",
            f"Alert Type:         {self.alert_type}",
            f"Risk Level:         {self.risk_level}",
            f"Reason:             {self.reason}",
            f"Recommended Action: {self.recommended_action}",
        ]
        if self.overload_probability is not None:
            lines.append(f"Overload Prob:      {self.overload_probability * 100.0:.1f}%")
        if self.anomaly_score is not None:
            lines.append(f"Anomaly Risk Index: {self.anomaly_score:.3f}")
        lines.append("=" * 35)
        return "\n".join(lines)


class UnifiedAlertEngine:
    """
    Modular Alert Engine for multi-transformer monitoring.
    Evaluates streaming or batched telemetry records, combining:
    - Real-time measured load against rated thresholds
    - Supervised next-hour classification probability (T2)
    - Unsupervised anomaly score (T1)
    - Regression load forecast (T1, T2, T3)
    """

    def __init__(
        self,
        warning_load_pct: float = 80.0,
        critical_load_pct: float = 100.0,
        classifier_alert_threshold: float = 0.35,
        anomaly_alert_threshold: float = 0.50,
    ):
        self.warning_load_pct = float(warning_load_pct)
        self.critical_load_pct = float(critical_load_pct)
        self.classifier_alert_threshold = float(classifier_alert_threshold)
        self.anomaly_alert_threshold = float(anomaly_alert_threshold)

    def generate_alert(
        self,
        transformer_id: str,
        timestamp: str,
        current_load_pct: float,
        predicted_load_pct: Optional[float] = None,
        overload_probability: Optional[float] = None,
        is_anomaly: Optional[bool] = None,
        anomaly_score: Optional[float] = None,
    ) -> TransformerAlert:
        """
        Evaluates signals in strict priority hierarchy to produce an actionable alert:
        1. CURRENT_OVERLOAD (CRITICAL): Measured load >= 100%
        2. PREDICTED_OVERLOAD (CRITICAL / WARNING): Model forecasts impending overload
        3. ELEVATED_LOAD_WARNING (WARNING): Measured load in [80%, 100%)
        4. LOAD_ANOMALY (WARNING): Anomaly detected during non-overloaded operation
        5. NORMAL_OPERATION (NORMAL): Safe nominal operation
        """
        t_id = transformer_id.upper()
        curr_load = float(current_load_pct)
        pred_load = float(predicted_load_pct) if predicted_load_pct is not None else None
        p_prob = float(overload_probability) if overload_probability is not None else None
        anom_flag = bool(is_anomaly) if is_anomaly is not None else False
        a_score = float(anomaly_score) if anomaly_score is not None else None

        # Priority 1: Confirmed Current Overload (Hardware/telemetry measured)
        if curr_load >= self.critical_load_pct:
            return TransformerAlert(
                transformer_id=t_id,
                timestamp=str(timestamp),
                current_load_pct=curr_load,
                predicted_load_pct=pred_load,
                risk_level=AlertLevel.CRITICAL.value,
                alert_type=AlertType.CURRENT_OVERLOAD.value,
                reason=f"Transformer load ({curr_load:.1f}%) has exceeded rated capacity ({self.critical_load_pct:.0f}%).",
                recommended_action="Notify the substation operator for advisory review. Recommend manual feeder inspection and assessment of load redistribution options, following approved operating procedures.",
                overload_probability=p_prob,
                anomaly_score=a_score,
            )

        # Priority 2: Predicted Impending Overload (Early Warning)
        # Triggered if classifier probability is high or regression predicts next-hour overload
        predicted_overload_flag = False
        reasons = []

        if p_prob is not None and p_prob >= self.classifier_alert_threshold:
            predicted_overload_flag = True
            reasons.append(f"Classification model signals {p_prob*100.0:.1f}% probability of next-hour overload (threshold: {self.classifier_alert_threshold*100.0:.1f}%).")

        if pred_load is not None and pred_load >= self.critical_load_pct:
            predicted_overload_flag = True
            reasons.append(f"Regression model forecasts next-hour load of {pred_load:.1f}% exceeding rated capacity.")

        if predicted_overload_flag:
            # If both signals or extremely high probability, CRITICAL warning; otherwise WARNING
            risk = AlertLevel.CRITICAL.value if (p_prob is not None and p_prob >= 0.70) or (pred_load is not None and pred_load >= 105.0) else AlertLevel.WARNING.value
            return TransformerAlert(
                transformer_id=t_id,
                timestamp=str(timestamp),
                current_load_pct=curr_load,
                predicted_load_pct=pred_load,
                risk_level=risk,
                alert_type=AlertType.PREDICTED_OVERLOAD.value,
                reason=" ".join(reasons),
                recommended_action="Alert distribution control center for advisory monitoring. Recommend verifying feeder current trend and reviewing backup feeder capacity before peak surge hours.",
                overload_probability=p_prob,
                anomaly_score=a_score,
            )

        # Priority 3: Elevated Load Warning (Measured load in 80% to 100%)
        if curr_load >= self.warning_load_pct:
            return TransformerAlert(
                transformer_id=t_id,
                timestamp=str(timestamp),
                current_load_pct=curr_load,
                predicted_load_pct=pred_load,
                risk_level=AlertLevel.WARNING.value,
                alert_type=AlertType.ELEVATED_LOAD_WARNING.value,
                reason=f"Transformer load ({curr_load:.1f}%) is in high-stress operating zone [{self.warning_load_pct:.0f}%, {self.critical_load_pct:.0f}%).",
                recommended_action="Monitor winding and oil temperature rise; verify auxiliary cooling systems are active.",
                overload_probability=p_prob,
                anomaly_score=a_score,
            )

        # Priority 4: Load / Telemetry Anomaly (Unusual behavior detected by unsupervised model)
        if anom_flag or (a_score is not None and a_score >= self.anomaly_alert_threshold):
            return TransformerAlert(
                transformer_id=t_id,
                timestamp=str(timestamp),
                current_load_pct=curr_load,
                predicted_load_pct=pred_load,
                risk_level=AlertLevel.WARNING.value,
                alert_type=AlertType.LOAD_ANOMALY.value,
                reason=f"Anomaly detector flagged unusual operational telemetry (anomaly risk index: {a_score:.3f} if score available else 'flagged').",
                recommended_action="Verify telemetry sensor calibration and inspect for abnormal phase imbalance or sudden thermal gradient shift.",
                overload_probability=p_prob,
                anomaly_score=a_score,
            )

        # Priority 5: Normal Operation
        return TransformerAlert(
            transformer_id=t_id,
            timestamp=str(timestamp),
            current_load_pct=curr_load,
            predicted_load_pct=pred_load,
            risk_level=AlertLevel.NORMAL.value,
            alert_type=AlertType.NORMAL_OPERATION.value,
            reason=f"Transformer operating within safe nominal capacity limits ({curr_load:.1f}% load).",
            recommended_action="Routine automated telemetry logging; no operator intervention required.",
            overload_probability=p_prob,
            anomaly_score=a_score,
        )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        transformer_id: str,
        timestamp_col: str = "datetime",
        load_col: str = "load_pct",
        pred_load_col: Optional[str] = "predicted_load_pct",
        prob_col: Optional[str] = "overload_probability",
        anomaly_flag_col: Optional[str] = "is_anomaly",
        anomaly_score_col: Optional[str] = "anomaly_score",
    ) -> List[TransformerAlert]:
        """Processes a dataframe of telemetry readings and outputs structured TransformerAlerts."""
        alerts = []
        for _, row in df.iterrows():
            ts = str(row[timestamp_col])
            curr_l = float(row[load_col])
            pred_l = float(row[pred_load_col]) if (pred_load_col and pred_load_col in row and pd.notna(row[pred_load_col])) else None
            prob = float(row[prob_col]) if (prob_col and prob_col in row and pd.notna(row[prob_col])) else None
            anom = bool(row[anomaly_flag_col]) if (anomaly_flag_col and anomaly_flag_col in row and pd.notna(row[anomaly_flag_col])) else None
            a_score = float(row[anomaly_score_col]) if (anomaly_score_col and anomaly_score_col in row and pd.notna(row[anomaly_score_col])) else None

            alert = self.generate_alert(
                transformer_id=transformer_id,
                timestamp=ts,
                current_load_pct=curr_l,
                predicted_load_pct=pred_l,
                overload_probability=prob,
                is_anomaly=anom,
                anomaly_score=a_score,
            )
            alerts.append(alert)
        return alerts


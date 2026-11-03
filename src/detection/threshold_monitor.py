"""
Configurable Threshold Monitoring Module for Transformer Telemetry.
Primarily applied to Transformer T3 (which has no historical overloads) and
reusable across any transformer feeder for direct rule-based monitoring.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class MonitoringStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class ThresholdMonitor:
    """
    Configurable threshold monitor evaluating telemetry measurements against
    engineering operating limits.
    
    Default Limits:
    - NORMAL: load < 80%
    - WARNING: load in [80%, 100%)
    - CRITICAL: load >= 100%
    
    Supports custom load and temperature thresholds without hardcoding.
    """

    def __init__(
        self,
        transformer_id: str = "T3",
        warning_load_pct: float = 80.0,
        critical_load_pct: float = 100.0,
        warning_temp_C: Optional[float] = 75.0,
        critical_temp_C: Optional[float] = 90.0,
    ):
        self.transformer_id = transformer_id.upper()
        self.warning_load_pct = float(warning_load_pct)
        self.critical_load_pct = float(critical_load_pct)
        self.warning_temp_C = float(warning_temp_C) if warning_temp_C is not None else None
        self.critical_temp_C = float(critical_temp_C) if critical_temp_C is not None else None

    def evaluate_sample(
        self,
        load_pct: float,
        oil_temp_C: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single telemetry sample against configured limits.
        """
        status = MonitoringStatus.NORMAL
        reasons = []

        # Load check
        if load_pct >= self.critical_load_pct:
            status = MonitoringStatus.CRITICAL
            reasons.append(f"Load percentage ({load_pct:.1f}%) exceeds critical limit ({self.critical_load_pct:.1f}%).")
        elif load_pct >= self.warning_load_pct:
            status = MonitoringStatus.WARNING
            reasons.append(f"Load percentage ({load_pct:.1f}%) in warning range [{self.warning_load_pct:.1f}%, {self.critical_load_pct:.1f}%).")

        # Temperature check (if provided and worse than load status)
        if oil_temp_C is not None:
            if self.critical_temp_C and oil_temp_C >= self.critical_temp_C:
                status = MonitoringStatus.CRITICAL
                reasons.append(f"Oil temperature ({oil_temp_C:.1f} °C) exceeds critical limit ({self.critical_temp_C:.1f} °C).")
            elif self.warning_temp_C and oil_temp_C >= self.warning_temp_C:
                if status != MonitoringStatus.CRITICAL:
                    status = MonitoringStatus.WARNING
                reasons.append(f"Oil temperature ({oil_temp_C:.1f} °C) in warning range.")

        if status == MonitoringStatus.NORMAL:
            reasons.append(f"Operating nominal within safe threshold (< {self.warning_load_pct:.1f}%).")

        return {
            'transformer_id': self.transformer_id,
            'status': status.value,
            'load_pct': float(load_pct),
            'oil_temp_C': float(oil_temp_C) if oil_temp_C is not None else None,
            'reason': " ".join(reasons),
        }

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        load_col: str = "load_pct",
        temp_col: Optional[str] = "oil_temp_C",
    ) -> pd.DataFrame:
        """
        Evaluates an entire dataframe against configured limits.
        Returns a DataFrame with status and reason columns.
        """
        loads = df[load_col].values
        temps = df[temp_col].values if (temp_col and temp_col in df.columns) else None

        statuses = []
        for i in range(len(df)):
            l = loads[i]
            t = temps[i] if temps is not None else None
            res = self.evaluate_sample(l, t)
            statuses.append(res['status'])

        result_df = df.copy()
        result_df['threshold_status'] = statuses
        return result_df

    def get_status_distribution(
        self,
        df: pd.DataFrame,
        load_col: str = "load_pct"
    ) -> Dict[str, Any]:
        """
        Returns counts and percentages for each status level across a dataset.
        """
        evaluated = self.evaluate_dataframe(df, load_col=load_col)
        counts = evaluated['threshold_status'].value_counts().to_dict()
        total = len(df)
        
        return {
            'transformer_id': self.transformer_id,
            'total_samples': total,
            'normal_count': int(counts.get(MonitoringStatus.NORMAL.value, 0)),
            'normal_pct': float(counts.get(MonitoringStatus.NORMAL.value, 0) / total * 100.0) if total > 0 else 0.0,
            'warning_count': int(counts.get(MonitoringStatus.WARNING.value, 0)),
            'warning_pct': float(counts.get(MonitoringStatus.WARNING.value, 0) / total * 100.0) if total > 0 else 0.0,
            'critical_count': int(counts.get(MonitoringStatus.CRITICAL.value, 0)),
            'critical_pct': float(counts.get(MonitoringStatus.CRITICAL.value, 0) / total * 100.0) if total > 0 else 0.0,
            'configured_warning_threshold': self.warning_load_pct,
            'configured_critical_threshold': self.critical_load_pct,
        }


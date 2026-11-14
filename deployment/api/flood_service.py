"""
AAPADAMITRA Flood River Discharge Prediction Service
Executes 24-hour river discharge forecasting at gauging stations using the
candidate Random Forest regressor. Compares forecasts against operational persistence.
Operates exclusively under Demonstration Mode.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging

from deployment.api.schemas import (
    FloodPredictionRequest,
    FloodPredictionResponse,
    FloodPredictions,
    FloodReferencePerformanceMetrics,
    FloodModelInfoResponse
)
from deployment.api.model_loader import model_loader

logger = logging.getLogger("FloodService")

FLOOD_FEATURES: List[str] = [
    "q_t",
    "q_lag1",
    "q_lag2",
    "q_lag3",
    "zero_gauge",
    "catchment_sqkm",
    "doy",
    "month"
]


def classify_flood_severity(predicted_q: float, current_q: float, delta_pct: float) -> str:
    """Classifies river discharge into qualitative flood stage warning bands."""
    if predicted_q >= 10000.0 or delta_pct >= 50.0:
        return "Severe Inundation Risk (Discharge Surge >= 50% or >= 10,000 m³/s)"
    elif predicted_q >= 5000.0 or delta_pct >= 25.0:
        return "Flood Warning Stage (Discharge Surge >= 25% or >= 5,000 m³/s)"
    elif delta_pct >= 10.0:
        return "Moderate Warning (Rising Stage >= 10%)"
    else:
        return "Normal Flow Range (Within Historical Baseline)"


class FloodInferenceService:
    """Orchestrates feature preparation and model inference for river discharge."""

    @staticmethod
    def prepare_features(req: FloodPredictionRequest) -> pd.DataFrame:
        """Structures input features matching Phase 3 flood training schema."""
        now = datetime.now(timezone.utc)
        doy = req.day_of_year if req.day_of_year is not None else now.timetuple().tm_yday
        month = req.month if req.month is not None else now.month

        row = {
            "q_t": float(req.q_t),
            "q_lag1": float(req.q_lag1),
            "q_lag2": float(req.q_lag2),
            "q_lag3": float(req.q_lag3),
            "zero_gauge": float(req.zero_gauge_m),
            "catchment_sqkm": float(req.catchment_sqkm),
            "doy": int(doy),
            "month": int(month)
        }
        return pd.DataFrame([row])[FLOOD_FEATURES]

    def predict(self, req: FloodPredictionRequest) -> FloodPredictionResponse:
        """
        Executes 24-hour River Discharge Forecast.
        Enforces physical bounds (non-negative discharge) and compares with persistence.
        """
        bundle = model_loader.get_flood_bundle()
        if not bundle.is_ready:
            raise RuntimeError("Flood candidate model artifact is not loaded or ready.")

        X = self.prepare_features(req)
        raw_pred = float(bundle.model.predict(X)[0])
        predicted_q = max(0.0, round(raw_pred, 2))
        current_q = float(req.q_t)

        delta_q = round(predicted_q - current_q, 2)
        delta_pct = round((delta_q / max(current_q, 1.0)) * 100.0, 2)
        severity = classify_flood_severity(predicted_q, current_q, delta_pct)

        predictions = FloodPredictions(
            predicted_discharge_m3_s=predicted_q,
            persistence_discharge_m3_s=current_q,
            discharge_delta_m3_s=delta_q,
            discharge_delta_pct=delta_pct,
            estimated_flood_severity=severity
        )

        input_summary = {
            "station_name": req.station_name or "UNSPECIFIED",
            "river_name": req.river_name or "UNSPECIFIED",
            "q_t_m3_s": current_q,
            "q_lag1_m3_s": float(req.q_lag1),
            "q_lag2_m3_s": float(req.q_lag2),
            "q_lag3_m3_s": float(req.q_lag3),
            "zero_gauge_m": float(req.zero_gauge_m),
            "catchment_sqkm": float(req.catchment_sqkm),
            "day_of_year": int(X["doy"].iloc[0]),
            "month": int(X["month"].iloc[0])
        }

        return FloodPredictionResponse(
            status="SUCCESS",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            model_version="v3.0",
            forecast_horizon_hours=24,
            station_name=req.station_name,
            river_name=req.river_name,
            input_summary=input_summary,
            predictions=predictions,
            reference_performance=FloodReferencePerformanceMetrics(),
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    @staticmethod
    def get_model_info() -> FloodModelInfoResponse:
        """Returns scientific metadata, validation metrics, and limitations for Flood module."""
        return FloodModelInfoResponse(
            hazard="flood",
            model_name="flood_discharge_rf",
            version="v3.0",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            promotion_rationale="REJECTED: Candidate NSE (0.9189) does not beat operational persistence (0.9229).",
            lead_time_hours=24,
            architecture="RandomForestRegressor (n_estimators=100, max_depth=12, min_samples_leaf=4)",
            features_required=FLOOD_FEATURES,
            feature_count=len(FLOOD_FEATURES),
            training_split="Chronological temporal split (Train <= 2018, Test 2019-2023 across Bihar/NWIC basins)",
            validation_protocol="Strict chronological holdout with operational persistence benchmarking",
            test_metrics={
                "NSE": 0.9189,
                "KGE": 0.9394,
                "MAE_m3_s": 53.19,
                "RMSE_m3_s": 271.38
            },
            baseline_metrics={
                "1_day_persistence_NSE": 0.9229,
                "1_day_persistence_KGE": 0.9615,
                "1_day_persistence_MAE_m3_s": 47.15,
                "1_day_persistence_RMSE_m3_s": 264.47
            },
            known_limitations=[
                "Upstream dam reservoir gate operations and emergency releases are unobserved in telemetry.",
                "Candidate model slightly underperforms 1-day persistence baseline (NSE 0.9189 vs 0.9229).",
                "Extreme flash flood surges (>99th percentile) may show localized underestimation.",
                "Statutory river level forecasts are governed exclusively by CWC."
            ],
            statutory_authority="Central Water Commission (CWC) / National Water Informatics Centre (NWIC)"
        )


flood_service = FloodInferenceService()


"""
AAPADAMITRA Landslide Susceptibility Prediction Service
Executes calibrated terrain and hydrometeorological landslide susceptibility screening
using the candidate Calibrated Random Forest classifier.
Operates exclusively under Demonstration Mode.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging

from deployment.api.schemas import (
    LandslidePredictionRequest,
    LandslidePredictionResponse,
    LandslidePredictions,
    LandslideReferencePerformanceMetrics,
    LandslideModelInfoResponse
)
from deployment.api.model_loader import model_loader

logger = logging.getLogger("LandslideService")

LANDSLIDE_FEATURES: List[str] = [
    "elevation_dem_m",
    "slope_degrees",
    "aspect_degrees",
    "profile_curvature",
    "topographic_wetness_index",
    "soil_thickness_m",
    "antecedent_rainfall_3d_mm",
    "antecedent_rainfall_7d_mm",
    "soil_moisture_0_7cm_m3m3",
    "soil_moisture_rootzone_m3m3"
]


def classify_susceptibility_level(prob: float) -> str:
    """Classifies probability into standard qualitative hazard zones."""
    if prob < 0.25:
        return "Low Susceptibility (< 25% probability)"
    elif prob < 0.60:
        return "Moderate Susceptibility (25% - 60% probability)"
    elif prob < 0.85:
        return "High Susceptibility (60% - 85% probability)"
    else:
        return "Very High Susceptibility (>= 85% probability)"


def detect_trigger_factors(req: LandslidePredictionRequest) -> List[str]:
    """Identifies primary geomorphological and hydrometeorological hazard contributors."""
    factors = []
    if req.slope_degrees >= 30.0:
        factors.append(f"Steep terrain slope ({req.slope_degrees:.1f}° >= 30° threshold)")
    if req.antecedent_rainfall_3d_mm >= 100.0:
        factors.append(f"Intense 3-day antecedent rainfall ({req.antecedent_rainfall_3d_mm:.1f} mm >= 100 mm)")
    if req.antecedent_rainfall_7d_mm >= 200.0:
        factors.append(f"Severe 7-day cumulative rainfall ({req.antecedent_rainfall_7d_mm:.1f} mm >= 200 mm)")
    if req.soil_moisture_rootzone_m3m3 >= 0.35:
        factors.append(f"Near-saturated rootzone soil moisture ({req.soil_moisture_rootzone_m3m3:.2f} m³/m³)")
    if req.topographic_wetness_index >= 8.0:
        factors.append(f"High topographic convergence zone (TWI = {req.topographic_wetness_index:.1f})")
    if not factors:
        factors.append("Sub-critical terrain and moisture conditions; low immediate trigger potential.")
    return factors


class LandslideInferenceService:
    """Orchestrates feature preparation and model inference for landslide susceptibility."""

    @staticmethod
    def prepare_features(req: LandslidePredictionRequest) -> pd.DataFrame:
        """Structures input features matching Phase 3 landslide training schema."""
        row = {
            "elevation_dem_m": float(req.elevation_dem_m),
            "slope_degrees": float(req.slope_degrees),
            "aspect_degrees": float(req.aspect_degrees),
            "profile_curvature": float(req.profile_curvature),
            "topographic_wetness_index": float(req.topographic_wetness_index),
            "soil_thickness_m": float(req.soil_thickness_m),
            "antecedent_rainfall_3d_mm": float(req.antecedent_rainfall_3d_mm),
            "antecedent_rainfall_7d_mm": float(req.antecedent_rainfall_7d_mm),
            "soil_moisture_0_7cm_m3m3": float(req.soil_moisture_0_7cm_m3m3),
            "soil_moisture_rootzone_m3m3": float(req.soil_moisture_rootzone_m3m3)
        }
        return pd.DataFrame([row])[LANDSLIDE_FEATURES]

    def predict(self, req: LandslidePredictionRequest) -> LandslidePredictionResponse:
        """Executes Landslide Susceptibility Screening inference."""
        bundle = model_loader.get_landslide_bundle()
        if not bundle.is_ready:
            raise RuntimeError("Landslide candidate model artifact is not loaded or ready.")

        X = self.prepare_features(req)
        prob = float(bundle.model.predict_proba(X)[:, 1][0])
        prob = round(max(0.0, min(1.0, prob)), 4)

        level = classify_susceptibility_level(prob)
        is_alert = prob >= 0.60
        factors = detect_trigger_factors(req)

        predictions = LandslidePredictions(
            susceptibility_probability=prob,
            susceptibility_level=level,
            high_risk_alert=is_alert,
            critical_trigger_factors=factors
        )

        input_summary = {
            "location_name": req.location_name or "UNSPECIFIED",
            "latitude": req.latitude,
            "longitude": req.longitude,
            "elevation_dem_m": float(req.elevation_dem_m),
            "slope_degrees": float(req.slope_degrees),
            "aspect_degrees": float(req.aspect_degrees),
            "profile_curvature": float(req.profile_curvature),
            "topographic_wetness_index": float(req.topographic_wetness_index),
            "soil_thickness_m": float(req.soil_thickness_m),
            "antecedent_rainfall_3d_mm": float(req.antecedent_rainfall_3d_mm),
            "antecedent_rainfall_7d_mm": float(req.antecedent_rainfall_7d_mm),
            "soil_moisture_0_7cm_m3m3": float(req.soil_moisture_0_7cm_m3m3),
            "soil_moisture_rootzone_m3m3": float(req.soil_moisture_rootzone_m3m3)
        }

        return LandslidePredictionResponse(
            status="SUCCESS",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            model_version="v3.0",
            location_name=req.location_name,
            input_summary=input_summary,
            predictions=predictions,
            reference_performance=LandslideReferencePerformanceMetrics(),
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    @staticmethod
    def get_model_info() -> LandslideModelInfoResponse:
        """Returns scientific metadata, validation metrics, and limitations for Landslide module."""
        return LandslideModelInfoResponse(
            hazard="landslide",
            model_name="landslide_susceptibility",
            version="v3.0",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            promotion_rationale="REJECTED: Candidate PR-AUC (1.0) does not outperform baseline (1.0). Retained in research catalog.",
            architecture="CalibratedClassifierCV (RandomForestClassifier, n_estimators=150, sigmoid calibration, 3-fold cv)",
            features_required=LANDSLIDE_FEATURES,
            feature_count=len(LANDSLIDE_FEATURES),
            training_split="Geographic region split (Train: Himalayas & Peninsular Plateau; Test: Western Ghats)",
            validation_protocol="Strict geographic holdout generalization test on Western Ghats partition",
            test_metrics={
                "pr_auc": 1.0,
                "roc_auc": 1.0,
                "brier_score": 0.0002,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0
            },
            baseline_metrics={
                "slope_heuristic_pr_auc": 1.0,
                "climatology_pr_auc": 0.6528
            },
            known_limitations=[
                "Independent audit identified that 1.0 PR-AUC is an artifact of synthetic non-landslide background sampling.",
                "Static DEM does not capture micro-topographic road cuts, quarrying, and slope toe excavation.",
                "Sub-daily cloudburst events (<1 hour) may trigger shallow slips unobserved in daily ERA5-Land reanalysis.",
                "Statutory landslide early warnings in India are governed exclusively by Geological Survey of India (GSI)."
            ],
            statutory_authority="Geological Survey of India (GSI)"
        )


landslide_service = LandslideInferenceService()


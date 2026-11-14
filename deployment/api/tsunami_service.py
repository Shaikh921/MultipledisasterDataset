"""
AAPADAMITRA Tsunami Early Screening Prediction Service
Executes rapid seismic hypocenter and oceanographic bathymetry screening
using the candidate Random Forest classifier.
Operates exclusively under Demonstration Mode.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging

from deployment.api.schemas import (
    TsunamiPredictionRequest,
    TsunamiPredictionResponse,
    TsunamiPredictions,
    TsunamiReferencePerformanceMetrics,
    TsunamiModelInfoResponse
)
from deployment.api.model_loader import model_loader

logger = logging.getLogger("TsunamiService")

TSUNAMI_FEATURES: List[str] = [
    "magnitude",
    "focal_depth_km",
    "epicenter_latitude",
    "epicenter_longitude",
    "gebco_bathymetry_m",
    "distance_to_nearest_coast_km",
    "tsunami_potential_index",
    "is_shallow_crustal",
    "is_deep_mantle",
    "log_moment_per_depth"
]

# Validation-tuned decision threshold from Phase 3 standardized training (optimized for F2 recall)
VALIDATION_TUNED_THRESHOLD: float = 0.44


def evaluate_empirical_rule(mag: float, depth: float, bathy: float) -> bool:
    """Checks traditional NOAA/PTWC empirical criteria (Mw >= 7.0, depth <= 60km, oceanic bathymetry)."""
    return bool(mag >= 7.0 and depth <= 60.0 and bathy <= -100.0)


def classify_tsunami_category(prob: float, alert: bool) -> str:
    """Classifies tsunamigenic probability into screening advisory tiers."""
    if prob >= 0.70:
        return "High Tsunami Generation Threat (Immediate Coastal Attention Required)"
    elif alert:
        return "Tsunami Screening Advisory (Exceeds 0.44 Validation Early Warning Threshold)"
    elif prob >= 0.20:
        return "Guarded / Information Level (Low Probability, Monitor Hydrodynamic Stations)"
    else:
        return "Negligible Tsunami Potential (Deep focus or sub-critical seismic energy)"


class TsunamiInferenceService:
    """Orchestrates feature preparation and model inference for tsunami potential screening."""

    @staticmethod
    def prepare_features(req: TsunamiPredictionRequest) -> pd.DataFrame:
        """Computes derived seismic proxies matching Phase 3 training pipeline."""
        mag = float(req.magnitude)
        depth = float(req.focal_depth_km)
        lat = float(req.epicenter_latitude)
        lon = float(req.epicenter_longitude)
        bathy = float(req.gebco_bathymetry_m)
        dist = float(req.distance_to_nearest_coast_km)

        # Derived seismic kinematic proxies
        tpi = round(float(mag - np.log10(max(depth, 1.0))), 4)
        is_shallow = 1.0 if depth < 70.0 else 0.0
        is_deep = 1.0 if depth >= 300.0 else 0.0
        log_moment = round(float(mag / (depth + 1.0)), 4)

        row = {
            "magnitude": mag,
            "focal_depth_km": depth,
            "epicenter_latitude": lat,
            "epicenter_longitude": lon,
            "gebco_bathymetry_m": bathy,
            "distance_to_nearest_coast_km": dist,
            "tsunami_potential_index": tpi,
            "is_shallow_crustal": is_shallow,
            "is_deep_mantle": is_deep,
            "log_moment_per_depth": log_moment
        }
        return pd.DataFrame([row])[TSUNAMI_FEATURES]

    def predict(self, req: TsunamiPredictionRequest) -> TsunamiPredictionResponse:
        """Executes Tsunami Screening inference."""
        bundle = model_loader.get_tsunami_bundle()
        if not bundle.is_ready:
            raise RuntimeError("Tsunami candidate model artifact is not loaded or ready.")

        X = self.prepare_features(req)
        prob = float(bundle.model.predict_proba(X)[:, 1][0])
        prob = round(max(0.0, min(1.0, prob)), 4)

        is_alert = prob >= VALIDATION_TUNED_THRESHOLD
        empirical_alert = evaluate_empirical_rule(
            mag=float(req.magnitude),
            depth=float(req.focal_depth_km),
            bathy=float(req.gebco_bathymetry_m)
        )
        category = classify_tsunami_category(prob, is_alert)

        predictions = TsunamiPredictions(
            tsunamigenic_probability=prob,
            decision_threshold=VALIDATION_TUNED_THRESHOLD,
            tsunami_screening_alert=is_alert,
            screening_category=category,
            empirical_rule_triggered=empirical_alert
        )

        input_summary = {
            "event_id": req.event_id or "UNSPECIFIED",
            "magnitude_mw": float(req.magnitude),
            "focal_depth_km": float(req.focal_depth_km),
            "epicenter_latitude": float(req.epicenter_latitude),
            "epicenter_longitude": float(req.epicenter_longitude),
            "gebco_bathymetry_m": float(req.gebco_bathymetry_m),
            "distance_to_nearest_coast_km": float(req.distance_to_nearest_coast_km),
            "tsunami_potential_index": float(X["tsunami_potential_index"].iloc[0]),
            "is_shallow_crustal": bool(X["is_shallow_crustal"].iloc[0]),
            "is_deep_mantle": bool(X["is_deep_mantle"].iloc[0])
        }

        return TsunamiPredictionResponse(
            status="SUCCESS",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            model_version="v3.0",
            event_id=req.event_id,
            input_summary=input_summary,
            predictions=predictions,
            reference_performance=TsunamiReferencePerformanceMetrics(),
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    @staticmethod
    def get_model_info() -> TsunamiModelInfoResponse:
        """Returns scientific metadata, validation metrics, and limitations for Tsunami module."""
        return TsunamiModelInfoResponse(
            hazard="tsunami",
            model_name="tsunami_classifier_rf",
            version="v3.0",
            execution_mode="DEMONSTRATION",
            production_status="NOT PROMOTED",
            demonstration_status="DEMONSTRATION_ACTIVE",
            promotion_rationale="RETAINED_EXISTING: Candidate PR-AUC (0.65) lower than existing baseline (0.6845).",
            architecture="RandomForestClassifier (n_estimators=100, max_depth=5, min_samples_leaf=3, class_weight=balanced)",
            features_required=TSUNAMI_FEATURES,
            feature_count=len(TSUNAMI_FEATURES),
            training_split="3-way chronological split (Train <= 2005; Val 2006-2015; Test 2016-2026)",
            validation_protocol="Threshold tuned on validation set for high F2 recall (0.44 threshold); tested on untouched holdout",
            test_metrics={
                "pr_auc": 0.6500,
                "roc_auc": 0.7708,
                "brier_score": 0.1506,
                "validation_tuned_threshold": 0.44,
                "precision": 1.0,
                "recall": 0.25,
                "f1": 0.40
            },
            baseline_metrics={
                "empirical_rule_pr_auc": 0.6458,
                "pre_existing_model_pr_auc": 0.6845
            },
            known_limitations=[
                "Point-source hypocentral parameters do not capture fault rupture geometry (slip strike/dip/rake) or dynamic seafloor deformation.",
                "Candidate PR-AUC (0.6500) did not exceed pre-existing benchmark (0.6845); retained in research catalog.",
                "Sample size of major modern tsunamis in the Indian Ocean is statistically very small.",
                "Official tsunami warnings in India are governed exclusively by INCOIS - ITEWC."
            ],
            statutory_authority="Indian National Centre for Ocean Information Services (INCOIS)"
        )


tsunami_service = TsunamiInferenceService()


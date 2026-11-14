"""
AAPADAMITRA Cyclone Prediction Service
Computes derived kinematic and thermodynamic features, invokes LightGBM models,
classifies IMD intensity stages, and structures operational forecasts.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import logging

from deployment.api.schemas import (
    CyclonePredictionRequest,
    CyclonePredictionResponse,
    CyclonePredictions,
    ReferencePerformanceMetrics,
    ModelInfoResponse
)
from deployment.api.model_loader import model_loader

logger = logging.getLogger("CycloneService")

# Exact feature order required by LightGBM model artifacts
ANTECEDENT_FEATURES: List[str] = [
    "current_latitude",
    "current_longitude",
    "current_wind_speed_kts_3min",
    "current_central_pressure_hpa",
    "dist2land_km",
    "translation_speed_kts",
    "heading_degrees",
    "sin_heading",
    "cos_heading",
    "sea_surface_temperature_c",
    "vertical_wind_shear_kts",
    "relative_humidity_700hpa_pct",
    "sst_above_26",
    "shear_favorable",
    "oceanic_heat_proxy",
    "pressure_wind_ratio",
    "coriolis_param"
]


def classify_imd_intensity(wind_kts: float) -> str:
    """Classifies sustained wind speed (3-min) into statutory IMD cyclone stages."""
    if wind_kts < 17.0:
        return "Low Pressure Area (<17 kts)"
    elif wind_kts <= 27.0:
        return "Depression (17-27 kts)"
    elif wind_kts <= 33.0:
        return "Deep Depression (28-33 kts)"
    elif wind_kts <= 47.0:
        return "Cyclonic Storm (34-47 kts)"
    elif wind_kts <= 63.0:
        return "Severe Cyclonic Storm (48-63 kts)"
    elif wind_kts <= 89.0:
        return "Very Severe Cyclonic Storm (64-89 kts)"
    elif wind_kts <= 119.0:
        return "Extremely Severe Cyclonic Storm (90-119 kts)"
    else:
        return "Super Cyclonic Storm (>=120 kts)"


class CycloneInferenceService:
    """Orchestrates feature preparation and model inference."""

    @staticmethod
    def prepare_features(req: CyclonePredictionRequest) -> pd.DataFrame:
        """Derives kinematic and thermodynamic features matching Phase 3 schema."""
        lat = float(req.current_latitude)
        lon = float(req.current_longitude)
        wind = float(req.current_wind_speed_kts_3min)
        pres = float(req.current_central_pressure_hpa)
        dist = float(req.dist2land_km)
        spd = float(req.translation_speed_kts)
        hdg = float(req.heading_degrees)
        sst = float(req.sea_surface_temperature_c)
        shear = float(req.vertical_wind_shear_kts)
        rh = float(req.relative_humidity_700hpa_pct)

        # Derived Kinematic and Thermodynamic Features
        sin_heading = round(float(np.sin(np.radians(hdg))), 4)
        cos_heading = round(float(np.cos(np.radians(hdg))), 4)
        sst_above_26 = round(float(max(0.0, sst - 26.0)), 2)
        shear_favorable = int(shear < 15.0)
        oceanic_heat_proxy = round(float(sst_above_26 * (1.0 if dist > 100.0 else 0.0)), 2)
        pressure_wind_ratio = round(float(pres / (wind + 1.0)), 3)
        coriolis_param = round(float(2.0 * 7.2921e-5 * np.sin(np.radians(lat)) * 1e4), 4)

        row = {
            "current_latitude": lat,
            "current_longitude": lon,
            "current_wind_speed_kts_3min": wind,
            "current_central_pressure_hpa": pres,
            "dist2land_km": dist,
            "translation_speed_kts": spd,
            "heading_degrees": hdg,
            "sin_heading": sin_heading,
            "cos_heading": cos_heading,
            "sea_surface_temperature_c": sst,
            "vertical_wind_shear_kts": shear,
            "relative_humidity_700hpa_pct": rh,
            "sst_above_26": sst_above_26,
            "shear_favorable": shear_favorable,
            "oceanic_heat_proxy": oceanic_heat_proxy,
            "pressure_wind_ratio": pressure_wind_ratio,
            "coriolis_param": coriolis_param
        }

        # Build single-row DataFrame with exact columns
        df = pd.DataFrame([row], columns=ANTECEDENT_FEATURES)
        return df

    def predict(self, req: CyclonePredictionRequest) -> CyclonePredictionResponse:
        """Executes multi-task 24-hour prediction."""
        bundle = model_loader.get_bundle()
        if not bundle.is_ready:
            raise RuntimeError("Cyclone model bundle is not loaded or missing components.")

        X = self.prepare_features(req)

        # 1. Regress Displacements & Intensity
        dlat_pred = float(bundle.model_lat.predict(X)[0])
        dlon_pred = float(bundle.model_lon.predict(X)[0])
        dwind_pred = float(bundle.model_wind.predict(X)[0])

        # 2. Predict Rapid Intensification Probability
        prob_ri = float(bundle.model_ri.predict(X)[0])
        ri_alert = bool(prob_ri >= 0.35)

        # 3. Compute absolute predicted values
        pred_lat = round(float(req.current_latitude + dlat_pred), 3)
        pred_lon = round(float(req.current_longitude + dlon_pred), 3)
        pred_wind = round(float(max(15.0, req.current_wind_speed_kts_3min + dwind_pred)), 1)
        imd_stage = classify_imd_intensity(pred_wind)

        predictions = CyclonePredictions(
            predicted_latitude=pred_lat,
            predicted_longitude=pred_lon,
            delta_latitude=round(dlat_pred, 3),
            delta_longitude=round(dlon_pred, 3),
            predicted_wind_speed_kts=pred_wind,
            delta_wind_speed_kts=round(dwind_pred, 1),
            imd_intensity_category=imd_stage,
            rapid_intensification_probability=round(prob_ri, 4),
            rapid_intensification_alert=ri_alert
        )

        input_summary = {
            "current_latitude": req.current_latitude,
            "current_longitude": req.current_longitude,
            "current_wind_speed_kts_3min": req.current_wind_speed_kts_3min,
            "current_central_pressure_hpa": req.current_central_pressure_hpa,
            "dist2land_km": req.dist2land_km,
            "heading_degrees": req.heading_degrees,
            "translation_speed_kts": req.translation_speed_kts,
            "sea_surface_temperature_c": req.sea_surface_temperature_c,
            "vertical_wind_shear_kts": req.vertical_wind_shear_kts
        }

        return CyclonePredictionResponse(
            status="SUCCESS",
            model_version="v3.0",
            forecast_horizon_hours=24,
            storm_name=req.storm_name,
            input_summary=input_summary,
            predictions=predictions,
            reference_performance=ReferencePerformanceMetrics(),
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns metadata card for the promoted Cyclone model."""
        return ModelInfoResponse(
            hazard="cyclone",
            model_name="cyclone_track_and_intensity_24h",
            version="v3.0",
            production_status="PROMOTED TO PRODUCTION",
            promotion_rationale="PROMOTED: Track error (136.5km) outperforms persistence (161.84km).",
            lead_time_hours=24,
            architecture="LightGBM Multi-Task Regressors (lat, lon, intensity) + Binary Classifier (RI)",
            features_required=ANTECEDENT_FEATURES,
            feature_count=len(ANTECEDENT_FEATURES),
            training_split="Future-Period Holdout with Grouped Storms (GroupKFold on storm_sid; train < 2021, test >= 2021)",
            validation_protocol="Independent temporal holdout on 16 named NIO storms (132 six-hourly fixes)",
            test_metrics={
                "mean_track_error_km": 136.50,
                "median_track_error_km": 118.24,
                "max_track_error_km": 553.36,
                "intensity_mae_kts": 11.30,
                "intensity_rmse_kts": 14.82,
                "ri_pr_auc": 0.2379,
                "ri_roc_auc": 0.7412
            },
            baseline_metrics={
                "dead_reckoning_persistence_track_error_km": 161.84,
                "dead_reckoning_persistence_intensity_mae_kts": 14.50,
                "climatology_track_error_km": 215.40
            },
            known_limitations=[
                "Extratropical transitioning storms or asymmetric landfall shearing exhibit higher track error.",
                "Rapid intensification in the Bay of Bengal can be triggered by mesoscale convective bursts below 6-hour temporal sampling.",
                "Official cyclone warnings and landfalls are governed statutorily by IMD RSMC New Delhi."
            ],
            statutory_authority="India Meteorological Department (IMD) - RSMC New Delhi"
        )


cyclone_service = CycloneInferenceService()

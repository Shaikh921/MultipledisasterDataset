"""
AAPADAMITRA API Schemas
Pydantic v2 schemas for request validation, structured responses, model metadata,
and statutory disclaimers.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ------------------------------------------------------------------------------
# 1. CYCLONE PREDICTION SCHEMAS
# ------------------------------------------------------------------------------
class CyclonePredictionRequest(BaseModel):
    """
    Input schema for North Indian Ocean 24-hour Cyclone Forecast.
    All antecedent meteorological and kinematic observations must be provided.
    Trigonometric and derived proxies are computed server-side.
    """
    current_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Current storm center latitude in WGS84 degrees North (typically 0.0 to 35.0 for NIO)",
        examples=[18.8]
    )
    current_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Current storm center longitude in WGS84 degrees East (typically 45.0 to 105.0 for NIO)",
        examples=[89.4]
    )
    current_wind_speed_kts_3min: float = Field(
        ...,
        ge=10.0,
        le=200.0,
        description="Current 3-minute sustained maximum wind speed in knots (WMO/IMD standard)",
        examples=[32.4]
    )
    current_central_pressure_hpa: float = Field(
        ...,
        ge=850.0,
        le=1050.0,
        description="Estimated minimum central barometric pressure in hectopascals (hPa / mb)",
        examples=[993.0]
    )
    dist2land_km: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Great-circle distance to nearest major continental or island coastline in kilometers",
        examples=[314.0]
    )
    translation_speed_kts: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Forward storm motion translation velocity over the past 6 hours in knots",
        examples=[10.0]
    )
    heading_degrees: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Azimuthal direction of storm motion in degrees (0 = North, 90 = East, 180 = South, 270 = West)",
        examples=[315.0]
    )
    sea_surface_temperature_c: float = Field(
        ...,
        ge=10.0,
        le=40.0,
        description="Underlying sea surface temperature (SST) in degrees Celsius",
        examples=[29.4]
    )
    vertical_wind_shear_kts: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Environmental deep-layer vertical wind shear between 200 hPa and 850 hPa in knots",
        examples=[16.2]
    )
    relative_humidity_700hpa_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mid-tropospheric ambient relative humidity at 700 hPa isobaric level (percentage 0-100)",
        examples=[81.9]
    )
    storm_name: Optional[str] = Field(
        None,
        description="Optional operational storm identifier or named system (e.g. 'REMAL', 'MOCHA')",
        examples=["REMAL"]
    )
    timestamp_utc: Optional[str] = Field(
        None,
        description="Optional observation timestamp in ISO 8601 UTC format (YYYY-MM-DDTHH:MM:SSZ)",
        examples=["2024-05-25T12:00:00Z"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_latitude": 18.8,
                "current_longitude": 89.4,
                "current_wind_speed_kts_3min": 32.4,
                "current_central_pressure_hpa": 993.0,
                "dist2land_km": 314.0,
                "translation_speed_kts": 10.0,
                "heading_degrees": 315.0,
                "sea_surface_temperature_c": 29.4,
                "vertical_wind_shear_kts": 16.2,
                "relative_humidity_700hpa_pct": 81.9,
                "storm_name": "REMAL",
                "timestamp_utc": "2024-05-25T12:00:00Z"
            }
        }
    )


class ReferencePerformanceMetrics(BaseModel):
    """Historical holdout benchmark metrics for scientific context."""
    mean_track_error_km: float = Field(136.50, description="Mean great-circle geodesic track error at +24h lead")
    median_track_error_km: float = Field(118.24, description="Median track error at +24h lead")
    intensity_mae_kts: float = Field(11.30, description="Mean Absolute Error in 24h wind intensity forecast")
    intensity_rmse_kts: float = Field(14.82, description="Root Mean Squared Error in 24h wind intensity forecast")
    baseline_persistence_track_error_km: float = Field(161.84, description="Dead-reckoning persistence benchmark track error")
    rapid_intensification_pr_auc: float = Field(0.2379, description="PR-AUC on holdout (base rate: 0.0833)")
    performance_note: str = Field(
        "Metrics evaluated on independent GroupKFold holdout (2021-2024 North Indian Ocean fixes). Single-point predictions must not be interpreted as deterministic landfall tracks.",
        description="Contextual note"
    )


class CyclonePredictions(BaseModel):
    """Predicted coordinates, intensity and rapid intensification flags at +24h."""
    predicted_latitude: float = Field(..., description="Forecast storm center latitude at +24h (WGS84 degrees North)")
    predicted_longitude: float = Field(..., description="Forecast storm center longitude at +24h (WGS84 degrees East)")
    delta_latitude: float = Field(..., description="Predicted latitudinal displacement over 24h (degrees)")
    delta_longitude: float = Field(..., description="Predicted longitudinal displacement over 24h (degrees)")
    predicted_wind_speed_kts: float = Field(..., description="Forecast 3-minute sustained wind speed at +24h in knots")
    delta_wind_speed_kts: float = Field(..., description="Predicted wind speed change over 24h (knots)")
    imd_intensity_category: str = Field(..., description="Official IMD cyclone intensity stage classification")
    rapid_intensification_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of Rapid Intensification (+30 kts or +15.4 m/s within 24h)"
    )
    rapid_intensification_alert: bool = Field(
        ...,
        description="Operational RI alert flag triggered if probability >= 0.35 threshold"
    )


class CyclonePredictionResponse(BaseModel):
    """Complete structured response for 24-hour Cyclone Forecast."""
    status: str = Field("SUCCESS", description="Execution status")
    model_version: str = Field("v3.0", description="Production model registry version")
    forecast_horizon_hours: int = Field(24, description="Forecast lead time in hours")
    storm_name: Optional[str] = Field(None, description="Storm identifier if provided")
    input_summary: Dict[str, Any] = Field(..., description="Summary of input observations")
    predictions: CyclonePredictions = Field(..., description="Forecast outcomes at +24h")
    reference_performance: ReferencePerformanceMetrics = Field(
        default_factory=ReferencePerformanceMetrics,
        description="Historical validation performance context"
    )
    statutory_disclaimer: str = Field(
        "STATUTORY NOTICE: AAPADAMITRA tropical cyclone models are experimental scientific decision-support aids. Official cyclone tracks, intensity advisories, and landfall warnings for the North Indian Ocean are issued exclusively by the India Meteorological Department (IMD) - Regional Specialized Meteorological Centre (RSMC) New Delhi. Do NOT use this model for operational life-safety evacuation decisions without consulting official IMD bulletins.",
        description="Statutory early warning authority demarcation"
    )
    generated_at: str = Field(..., description="UTC ISO 8601 generation timestamp")


# ------------------------------------------------------------------------------
# 2. MODEL METADATA SCHEMAS
# ------------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    """Detailed model registry card and scientific metadata."""
    hazard: str = "cyclone"
    model_name: str = "cyclone_track_and_intensity_24h"
    version: str = "v3.0"
    production_status: str = "PROMOTED TO PRODUCTION"
    promotion_rationale: str = "PROMOTED: Track error (136.5km) outperforms persistence (161.84km)."
    lead_time_hours: int = 24
    architecture: str = "Multi-task LightGBM Regressors (dlat, dlon, dwind) + LightGBM Classifier (RI)"
    features_required: List[str]
    feature_count: int
    training_split: str
    validation_protocol: str
    test_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    known_limitations: List[str]
    statutory_authority: str = "India Meteorological Department (IMD) - RSMC New Delhi"


# ------------------------------------------------------------------------------
# 3. HEALTH & MULTI-HAZARD STATUS SCHEMAS
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """System health, memory consumption, and model readiness status."""
    status: str
    service: str = "AAPADAMITRA Multi-Hazard Inference API"
    environment: str
    uptime_seconds: float
    memory_rss_mb: float
    memory_limit_mb: float = 512.0
    memory_usage_pct: float
    cyclone_model_loaded: bool
    model_version: str = "v3.0"
    active_hazard: str = "cyclone"
    timestamp_utc: str


class HazardStatus(BaseModel):
    """Hazard deployment status and promotion rationale."""
    hazard: str
    model_name: str
    version: str
    production_status: str
    is_active: bool
    promotion_rationale: str
    available_endpoints: List[str]


class ModelsStatusResponse(BaseModel):
    """Multi-hazard deployment registry showing active vs non-promoted models."""
    active_hazard_count: int
    total_hazards: int = 4
    models: Dict[str, HazardStatus]
    system_note: str = (
        "Strictly following Phase 3 scientific audit decisions: Only the Cyclone model "
        "is promoted to production. Flood, Landslide, and Tsunami endpoints are unpromoted "
        "and retained in the research catalog."
    )

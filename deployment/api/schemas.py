"""
AAPADAMITRA API Schemas
Pydantic v2 schemas for request validation, structured responses, model metadata,
and statutory disclaimers across all four natural disaster hazard modules:
1. Tropical Cyclone (Track, Intensity & Rapid Intensification)
2. River Flood (Gauging Station Discharge Forecasting)
3. Landslide Susceptibility (Terrain & Hydrometeorological Screening)
4. Tsunami Potential (Seismic Hypocenter & Bathymetric Early Screening)
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ------------------------------------------------------------------------------
# 1. CYCLONE PREDICTION SCHEMAS (PRODUCTION PROMOTED)
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


CycloneReferencePerformanceMetrics = ReferencePerformanceMetrics


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
    execution_mode: str = Field("PRODUCTION", description="Operational execution tier")
    production_status: str = Field("PROMOTED TO PRODUCTION", description="Model production validation status")
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
# 2. FLOOD DISCHARGE SCHEMAS (DEMONSTRATION ONLY)
# ------------------------------------------------------------------------------
class FloodPredictionRequest(BaseModel):
    """
    Input schema for 24-hour River Discharge Forecast (Gauging Station).
    All discharge inputs are in cubic meters per second (m^3/s).
    """
    q_t: float = Field(
        ...,
        ge=0.0,
        le=150000.0,
        description="Current observed river discharge at gauging station in m^3/s",
        examples=[1250.0]
    )
    q_lag1: float = Field(
        ...,
        ge=0.0,
        le=150000.0,
        description="Discharge 1 day prior (t-1) in m^3/s",
        examples=[1180.0]
    )
    q_lag2: float = Field(
        ...,
        ge=0.0,
        le=150000.0,
        description="Discharge 2 days prior (t-2) in m^3/s",
        examples=[1100.0]
    )
    q_lag3: float = Field(
        ...,
        ge=0.0,
        le=150000.0,
        description="Discharge 3 days prior (t-3) in m^3/s",
        examples=[1050.0]
    )
    zero_gauge_m: float = Field(
        50.0,
        ge=-50.0,
        le=5000.0,
        description="Station zero gauge datum elevation above mean sea level in meters",
        examples=[50.0]
    )
    catchment_sqkm: float = Field(
        10000.0,
        ge=10.0,
        le=2000000.0,
        description="Drainage basin catchment area in square kilometers",
        examples=[10000.0]
    )
    day_of_year: Optional[int] = Field(
        None,
        ge=1,
        le=366,
        description="Day of the year (1-366). Defaults to current UTC day of year if omitted",
        examples=[205]
    )
    month: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="Calendar month (1-12). Defaults to current UTC month if omitted",
        examples=[7]
    )
    station_name: Optional[str] = Field(
        None,
        description="Optional gauging station name (e.g., 'Baltara', 'Patna')",
        examples=["Baltara"]
    )
    river_name: Optional[str] = Field(
        None,
        description="Optional river basin name (e.g., 'Kosi', 'Ganga')",
        examples=["Kosi"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "station_name": "Baltara",
                "river_name": "Kosi",
                "q_t": 1250.0,
                "q_lag1": 1180.0,
                "q_lag2": 1100.0,
                "q_lag3": 1050.0,
                "zero_gauge_m": 33.85,
                "catchment_sqkm": 61000.0,
                "day_of_year": 205,
                "month": 7
            }
        }
    )


class FloodReferencePerformanceMetrics(BaseModel):
    """Historical validation performance context for Flood Candidate model."""
    nse: float = Field(0.9189, description="Nash-Sutcliffe Efficiency on holdout test set")
    kge: float = Field(0.9394, description="Kling-Gupta Efficiency on holdout test set")
    mae_m3_s: float = Field(53.19, description="Mean Absolute Error in m^3/s")
    rmse_m3_s: float = Field(271.38, description="Root Mean Squared Error in m^3/s")
    baseline_persistence_nse: float = Field(0.9229, description="1-day persistence benchmark NSE")
    performance_caveat: str = Field(
        "Candidate Random Forest model achieved NSE 0.9189, which did not beat 1-day persistence (0.9229). Retained for demonstration under research catalog.",
        description="Scientific audit limitation notice"
    )


class FloodPredictions(BaseModel):
    """Predicted river discharge and severity indicators at +24h lead time."""
    predicted_discharge_m3_s: float = Field(..., description="Forecast discharge at +24h lead time in m^3/s")
    persistence_discharge_m3_s: float = Field(..., description="Operational 1-day persistence benchmark (q_t) in m^3/s")
    discharge_delta_m3_s: float = Field(..., description="Predicted change in discharge over 24h in m^3/s")
    discharge_delta_pct: float = Field(..., description="Percentage change in discharge relative to q_t")
    estimated_flood_severity: str = Field(..., description="Severity classification based on discharge hydrograph")


class FloodPredictionResponse(BaseModel):
    """Complete structured response for 24-hour River Discharge Forecast."""
    status: str = Field("SUCCESS", description="Execution status")
    execution_mode: str = Field("DEMONSTRATION", description="Execution operational tier")
    production_status: str = Field("NOT PROMOTED", description="Model production validation status")
    demonstration_status: str = Field("DEMONSTRATION_ACTIVE", description="Demonstration mode availability")
    model_version: str = Field("v3.0", description="Model version")
    forecast_horizon_hours: int = Field(24, description="Forecast lead time in hours")
    station_name: Optional[str] = Field(None, description="Station identifier")
    river_name: Optional[str] = Field(None, description="River basin name")
    input_summary: Dict[str, Any] = Field(..., description="Summary of input observations")
    predictions: FloodPredictions = Field(..., description="Forecast discharge outcomes")
    reference_performance: FloodReferencePerformanceMetrics = Field(
        default_factory=FloodReferencePerformanceMetrics,
        description="Scientific benchmark metrics"
    )
    scientific_caveat: str = Field(
        "DEMONSTRATION MODE NOTICE: This candidate Random Forest model did not outperform operational persistence in Phase 3 benchmarks (Candidate NSE 0.9189 vs Persistence 0.9229). It is provided strictly for technical demonstration and educational research. Upstream dam operations and reservoir releases are not captured in telemetry.",
        description="Scientific caveat"
    )
    statutory_disclaimer: str = Field(
        "STATUTORY NOTICE: AAPADAMITRA river discharge predictions are experimental scientific decision-support aids. Official flood forecasts, stage level hydrographs, and inundation warnings in India are issued exclusively by the Central Water Commission (CWC) and National Water Informatics Centre (NWIC). Do NOT use this model for operational life-safety or evacuation decisions.",
        description="Statutory authority notice"
    )
    generated_at: str = Field(..., description="UTC ISO 8601 generation timestamp")


class FloodModelInfoResponse(BaseModel):
    """Detailed model registry card and scientific metadata for Flood module."""
    hazard: str = "flood"
    model_name: str = "flood_discharge_rf"
    version: str = "v3.0"
    execution_mode: str = "DEMONSTRATION"
    production_status: str = "NOT PROMOTED"
    demonstration_status: str = "DEMONSTRATION_ACTIVE"
    promotion_rationale: str = "REJECTED: Candidate NSE (0.9189) does not beat operational persistence (0.9229)."
    lead_time_hours: int = 24
    architecture: str = "RandomForestRegressor (100 estimators, max_depth=12, min_samples_leaf=4)"
    features_required: List[str]
    feature_count: int
    training_split: str
    validation_protocol: str
    test_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    known_limitations: List[str]
    statutory_authority: str = "Central Water Commission (CWC) / National Water Informatics Centre (NWIC)"


# ------------------------------------------------------------------------------
# 3. LANDSLIDE SUSCEPTIBILITY SCHEMAS (DEMONSTRATION ONLY)
# ------------------------------------------------------------------------------
class LandslidePredictionRequest(BaseModel):
    """
    Input schema for Landslide Susceptibility Screening.
    Accepts terrain morphometry (DEM-derived) and hydrometeorological moisture indicators.
    """
    elevation_dem_m: float = Field(
        ...,
        ge=-100.0,
        le=9000.0,
        description="Digital Elevation Model (SRTM/Copernicus) altitude in meters",
        examples=[1240.0]
    )
    slope_degrees: float = Field(
        ...,
        ge=0.0,
        le=90.0,
        description="Topographic terrain slope in degrees (0-90)",
        examples=[32.5]
    )
    aspect_degrees: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Slope aspect azimuth in degrees (0=North, 90=East, 180=South, 270=West)",
        examples=[180.0]
    )
    profile_curvature: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="Surface profile curvature (negative=convex, positive=concave)",
        examples=[-0.05]
    )
    topographic_wetness_index: float = Field(
        ...,
        ge=0.0,
        le=35.0,
        description="Topographic Wetness Index ln(a / tan(beta))",
        examples=[6.2]
    )
    soil_thickness_m: float = Field(
        1.5,
        ge=0.0,
        le=25.0,
        description="Estimated regolith / soil mantle depth in meters",
        examples=[1.5]
    )
    antecedent_rainfall_3d_mm: float = Field(
        ...,
        ge=0.0,
        le=2000.0,
        description="3-day cumulative antecedent rainfall in mm",
        examples=[125.0]
    )
    antecedent_rainfall_7d_mm: float = Field(
        ...,
        ge=0.0,
        le=4000.0,
        description="7-day cumulative antecedent rainfall in mm",
        examples=[210.0]
    )
    soil_moisture_0_7cm_m3m3: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Volumetric soil water content in topsoil (0-7 cm) in m^3/m^3 (0.0 to 1.0)",
        examples=[0.42]
    )
    soil_moisture_rootzone_m3m3: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Volumetric soil water content in rootzone (7-100 cm) in m^3/m^3 (0.0 to 1.0)",
        examples=[0.38]
    )
    location_name: Optional[str] = Field(
        None,
        description="Optional geographical place or hill tract name (e.g. 'Wayanad Hills', 'Idukki')",
        examples=["Wayanad Hills"]
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Optional WGS84 latitude coordinate",
        examples=[11.68]
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Optional WGS84 longitude coordinate",
        examples=[76.13]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "location_name": "Wayanad Hills",
                "latitude": 11.68,
                "longitude": 76.13,
                "elevation_dem_m": 1240.0,
                "slope_degrees": 32.5,
                "aspect_degrees": 180.0,
                "profile_curvature": -0.05,
                "topographic_wetness_index": 6.2,
                "soil_thickness_m": 1.5,
                "antecedent_rainfall_3d_mm": 125.0,
                "antecedent_rainfall_7d_mm": 210.0,
                "soil_moisture_0_7cm_m3m3": 0.42,
                "soil_moisture_rootzone_m3m3": 0.38
            }
        }
    )


class LandslideReferencePerformanceMetrics(BaseModel):
    """Historical validation performance context for Landslide Candidate model."""
    pr_auc: float = Field(1.0, description="Precision-Recall AUC on Western Ghats holdout partition")
    roc_auc: float = Field(1.0, description="ROC-AUC on Western Ghats holdout partition")
    brier_score: float = Field(0.0002, description="Brier calibration score")
    baseline_heuristic_pr_auc: float = Field(1.0, description="Slope-heuristic baseline PR-AUC")
    audit_caveat: str = Field(
        "Candidate model exhibits 1.0 PR-AUC due to synthetic non-landslide background sampling separation. Not approved for production; enabled for demonstration.",
        description="Scientific audit finding"
    )


class LandslidePredictions(BaseModel):
    """Predicted susceptibility probability and terrain risk indicators."""
    susceptibility_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated landslide probability (0.0 to 1.0)")
    susceptibility_level: str = Field(..., description="Qualitative susceptibility classification")
    high_risk_alert: bool = Field(..., description="High susceptibility alert flag (probability >= 0.60)")
    critical_trigger_factors: List[str] = Field(..., description="Key environmental/terrain contributing factors detected")


class LandslidePredictionResponse(BaseModel):
    """Complete structured response for Landslide Susceptibility Screening."""
    status: str = Field("SUCCESS", description="Execution status")
    execution_mode: str = Field("DEMONSTRATION", description="Execution operational tier")
    production_status: str = Field("NOT PROMOTED", description="Model production validation status")
    demonstration_status: str = Field("DEMONSTRATION_ACTIVE", description="Demonstration mode availability")
    model_version: str = Field("v3.0", description="Model version")
    location_name: Optional[str] = Field(None, description="Location identifier")
    input_summary: Dict[str, Any] = Field(..., description="Summary of input terrain and hydrometeorological features")
    predictions: LandslidePredictions = Field(..., description="Susceptibility prediction outcomes")
    reference_performance: LandslideReferencePerformanceMetrics = Field(
        default_factory=LandslideReferencePerformanceMetrics,
        description="Scientific benchmark context"
    )
    scientific_caveat: str = Field(
        "DEMONSTRATION MODE NOTICE: Scientific audit revealed artificial class separation from synthetic background points (1.0 PR-AUC artifact). Micro-topographic modifications (road cuts, toe excavation) and localized cloudbursts (<1 hour) are not captured in static DEM or daily ERA5-Land reanalysis. Retained strictly for demonstration.",
        description="Scientific caveat"
    )
    statutory_disclaimer: str = Field(
        "STATUTORY NOTICE: AAPADAMITRA landslide susceptibility predictions are experimental scientific decision-support aids. Official landslide warnings, hazard zonation maps, and slope stability bulletins in India are issued exclusively by the Geological Survey of India (GSI). Do NOT use this model for operational evacuation or construction safety decisions.",
        description="Statutory authority notice"
    )
    generated_at: str = Field(..., description="UTC ISO 8601 generation timestamp")


class LandslideModelInfoResponse(BaseModel):
    """Detailed model registry card and scientific metadata for Landslide module."""
    hazard: str = "landslide"
    model_name: str = "landslide_susceptibility"
    version: str = "v3.0"
    execution_mode: str = "DEMONSTRATION"
    production_status: str = "NOT PROMOTED"
    demonstration_status: str = "DEMONSTRATION_ACTIVE"
    promotion_rationale: str = "REJECTED: Candidate PR-AUC (1.0) does not outperform baseline (1.0). Retained in research catalog."
    architecture: str = "CalibratedClassifierCV (RandomForestClassifier, n_estimators=150, sigmoid calibration, 3-fold cv)"
    features_required: List[str]
    feature_count: int
    training_split: str
    validation_protocol: str
    test_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    known_limitations: List[str]
    statutory_authority: str = "Geological Survey of India (GSI)"


# ------------------------------------------------------------------------------
# 4. TSUNAMI SCREENING SCHEMAS (DEMONSTRATION ONLY)
# ------------------------------------------------------------------------------
class TsunamiPredictionRequest(BaseModel):
    """
    Input schema for Tsunami Potential Early Screening.
    Accepts earthquake hypocenter parameters and coastal/bathymetric context.
    Server computes derived seismic indices and crustal classification.
    """
    magnitude: float = Field(
        ...,
        ge=4.0,
        le=10.0,
        description="Earthquake moment magnitude (Mw)",
        examples=[8.2]
    )
    focal_depth_km: float = Field(
        ...,
        ge=0.0,
        le=700.0,
        description="Focal hypocenter depth below seafloor in kilometers",
        examples=[15.0]
    )
    epicenter_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Epicenter latitude in WGS84 degrees North",
        examples=[3.3]
    )
    epicenter_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Epicenter longitude in WGS84 degrees East",
        examples=[95.8]
    )
    gebco_bathymetry_m: float = Field(
        ...,
        ge=-12000.0,
        le=100.0,
        description="Seafloor bathymetry depth at epicenter in meters (negative value for ocean)",
        examples=[-3500.0]
    )
    distance_to_nearest_coast_km: float = Field(
        ...,
        ge=0.0,
        le=20000.0,
        description="Great-circle distance to nearest coastline in kilometers",
        examples=[45.0]
    )
    event_id: Optional[str] = Field(
        None,
        description="Optional seismic event identifier (e.g. USGS ID 'us7000xxxx')",
        examples=["sumatra_2004_screening_analogue"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_id": "sumatra_2004_screening_analogue",
                "magnitude": 8.2,
                "focal_depth_km": 15.0,
                "epicenter_latitude": 3.3,
                "epicenter_longitude": 95.8,
                "gebco_bathymetry_m": -3500.0,
                "distance_to_nearest_coast_km": 45.0
            }
        }
    )


class TsunamiReferencePerformanceMetrics(BaseModel):
    """Historical validation performance context for Tsunami Candidate model."""
    pr_auc: float = Field(0.6500, description="Precision-Recall AUC on chronological holdout test set (2016-2026)")
    roc_auc: float = Field(0.7708, description="ROC-AUC on holdout test set")
    brier_score: float = Field(0.1506, description="Brier calibration score")
    baseline_empirical_pr_auc: float = Field(0.6458, description="Empirical magnitude-depth rule PR-AUC")
    existing_model_pr_auc: float = Field(0.6845, description="Pre-existing model PR-AUC benchmark")
    performance_caveat: str = Field(
        "Candidate Random Forest model achieved PR-AUC 0.6500, which is lower than the pre-existing baseline (0.6845). Retained for demonstration under research catalog.",
        description="Scientific audit finding"
    )


class TsunamiPredictions(BaseModel):
    """Predicted tsunamigenic probability and early screening classifications."""
    tsunamigenic_probability: float = Field(..., ge=0.0, le=1.0, description="Model-predicted probability of tsunamigenic potential (0.0 to 1.0)")
    decision_threshold: float = Field(0.44, description="Validation-tuned decision threshold (optimized for high F2 recall)")
    tsunami_screening_alert: bool = Field(..., description="Screening alert triggered if probability >= decision_threshold")
    screening_category: str = Field(..., description="Tsunami potential qualitative category")
    empirical_rule_triggered: bool = Field(..., description="Traditional empirical rule triggered (Mw >= 7.0, depth <= 60km, bathymetry <= -100m)")


class TsunamiPredictionResponse(BaseModel):
    """Complete structured response for Tsunami Potential Early Screening."""
    status: str = Field("SUCCESS", description="Execution status")
    execution_mode: str = Field("DEMONSTRATION", description="Execution operational tier")
    production_status: str = Field("NOT PROMOTED", description="Model production validation status")
    demonstration_status: str = Field("DEMONSTRATION_ACTIVE", description="Demonstration mode availability")
    model_version: str = Field("v3.0", description="Model version")
    event_id: Optional[str] = Field(None, description="Event identifier")
    input_summary: Dict[str, Any] = Field(..., description="Summary of input hypocentral and oceanographic parameters")
    predictions: TsunamiPredictions = Field(..., description="Screening prediction outcomes")
    reference_performance: TsunamiReferencePerformanceMetrics = Field(
        default_factory=TsunamiReferencePerformanceMetrics,
        description="Scientific benchmark metrics"
    )
    scientific_caveat: str = Field(
        "DEMONSTRATION MODE NOTICE: Candidate Random Forest screening model achieved PR-AUC 0.6500, lower than the pre-existing baseline (0.6845). Point-source hypocentral parameters do not capture fault geometry, focal mechanism (strike/dip/rake), or dynamic seafloor rupture displacement. The sample size for modern major tsunamis in the Indian Ocean basin is statistically small. Retained strictly for demonstration.",
        description="Scientific caveat"
    )
    statutory_disclaimer: str = Field(
        "STATUTORY NOTICE: AAPADAMITRA tsunami screening is an experimental scientific decision-support aid. Official tsunami warnings, watches, and advisories for the Indian Ocean are issued exclusively by the Indian National Centre for Ocean Information Services (INCOIS) - Indian Tsunami Early Warning Centre (ITEWC) and PTWC. Do NOT use this model for operational coastal evacuation decisions without official INCOIS notification.",
        description="Statutory authority notice"
    )
    generated_at: str = Field(..., description="UTC ISO 8601 generation timestamp")


class TsunamiModelInfoResponse(BaseModel):
    """Detailed model registry card and scientific metadata for Tsunami module."""
    hazard: str = "tsunami"
    model_name: str = "tsunami_classifier_rf"
    version: str = "v3.0"
    execution_mode: str = "DEMONSTRATION"
    production_status: str = "NOT PROMOTED"
    demonstration_status: str = "DEMONSTRATION_ACTIVE"
    promotion_rationale: str = "RETAINED_EXISTING: Candidate PR-AUC (0.65) lower than existing baseline (0.6845)."
    architecture: str = "RandomForestClassifier (100 estimators, max_depth=5, min_samples_leaf=3, class_weight=balanced)"
    features_required: List[str]
    feature_count: int
    training_split: str
    validation_protocol: str
    test_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    known_limitations: List[str]
    statutory_authority: str = "Indian National Centre for Ocean Information Services (INCOIS)"


# ------------------------------------------------------------------------------
# 5. CYCLONE MODEL INFO (PRESERVED FOR BACKWARD COMPATIBILITY)
# ------------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    """Detailed model registry card and scientific metadata for Cyclone v3.0."""
    hazard: str = "cyclone"
    model_name: str = "cyclone_track_and_intensity_24h"
    version: str = "v3.0"
    execution_mode: str = "PRODUCTION"
    production_status: str = "PROMOTED TO PRODUCTION"
    demonstration_status: str = "PRODUCTION_APPROVED"
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


CycloneModelInfoResponse = ModelInfoResponse


# ------------------------------------------------------------------------------
# 6. HEALTH & MULTI-HAZARD STATUS SCHEMAS
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
    flood_model_loaded: bool = False
    landslide_model_loaded: bool = False
    tsunami_model_loaded: bool = False
    demo_mode: bool = True
    model_version: str = "v3.0"
    active_hazard: str = "cyclone"
    timestamp_utc: str


class HazardStatus(BaseModel):
    """Hazard deployment status, promotion rationale, and demonstration availability."""
    hazard: str
    model_name: str
    version: str
    production_status: str
    is_active: bool
    demo_enabled: bool = True
    demo_status: str = "DEMONSTRATION_ONLY"
    promotion_rationale: str
    available_endpoints: List[str]
    statutory_authority: str = "Official National Agency"


class ModelsStatusResponse(BaseModel):
    """Multi-hazard deployment registry showing active vs demonstration models."""
    active_hazard_count: int
    demonstration_hazard_count: int = 4
    total_hazards: int = 4
    models: Dict[str, HazardStatus]
    system_note: str = (
        "Strictly following Phase 3 scientific audit decisions: Cyclone v3.0 is the single "
        "production-promoted model. Flood, Landslide, and Tsunami modules are operational "
        "under Demonstration Mode with explicit scientific caveats and statutory disclaimers."
    )


class HazardSummaryItem(BaseModel):
    """Summary item for GET /api/v1/hazards registry view."""
    hazard: str
    title: str
    model_name: str
    version: str
    production_status: str
    demo_enabled: bool
    demo_status: str
    promotion_rationale: str
    statutory_authority: str
    endpoints: Dict[str, str]
    key_metrics: Dict[str, Any]


class HazardsListResponse(BaseModel):
    """Complete catalog of hazards available across production and demonstration modes."""
    total_hazards: int = 4
    promoted_hazards_count: int = 1
    demonstration_hazards_count: int = 4
    operational_mode: str = "MULTI_HAZARD_DEMONSTRATION"
    hazards: List[HazardSummaryItem]
    system_note: str

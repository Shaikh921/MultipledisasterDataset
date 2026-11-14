"""
AAPADAMITRA Multi-Hazard AI Platform - Inference API Server
FastAPI application optimized for Render Free Web Service (512 MB RAM limit).
Hosts the production-promoted Cyclone v3.0 model while providing safe Demonstration
Mode access to Flood, Landslide, and Tsunami modules.
Loads model artifacts directly from the repository checkout with zero cloud storage dependencies.
"""

from typing import Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import os
import psutil
import logging

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from deployment.api.config import settings
from deployment.api.schemas import (
    # Cyclone
    CyclonePredictionRequest,
    CyclonePredictionResponse,
    ModelInfoResponse,
    # Flood
    FloodPredictionRequest,
    FloodPredictionResponse,
    FloodModelInfoResponse,
    # Landslide
    LandslidePredictionRequest,
    LandslidePredictionResponse,
    LandslideModelInfoResponse,
    # Tsunami
    TsunamiPredictionRequest,
    TsunamiPredictionResponse,
    TsunamiModelInfoResponse,
    # System & Registry
    HealthResponse,
    HazardStatus,
    ModelsStatusResponse,
    HazardsListResponse,
    HazardSummaryItem
)
from deployment.api.security import verify_api_key
from deployment.api.model_loader import model_loader
from deployment.api.cyclone_service import cyclone_service
from deployment.api.flood_service import flood_service
from deployment.api.landslide_service import landslide_service
from deployment.api.tsunami_service import tsunami_service

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AAPADAMITRA_API")

# Server start timestamp
SERVER_START_TIME = time.time()


def get_current_rss_mb() -> float:
    """Measures current process Resident Set Size (RSS) memory in MB."""
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception as e:
        logger.warning(f"Could not read process memory: {e}")
        return 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads and verifies models on startup. Raises fatal exception on failure.
    """
    logger.info("Initializing AAPADAMITRA Inference Service...")
    mem_before = get_current_rss_mb()
    logger.info(f"Memory RSS before model loading: {mem_before:.2f} MB")

    try:
        # Load Cyclone models directly from local repository (mandatory)
        cyclone_bundle = model_loader.load_cyclone_models()
        logger.info(f"Cyclone v3.0 loaded in {cyclone_bundle.load_time_seconds:.2f}s.")

        # In Demonstration Mode, load remaining three hazard models
        if settings.DEMO_MODE:
            logger.info("Demonstration mode active: Loading Flood, Landslide and Tsunami models...")
            loaded = model_loader.load_all_models()
            for h, b in loaded.items():
                if b and b.is_ready:
                    logger.info(f"  Hazard '{h}': Ready (load_time: {b.load_time_seconds:.3f}s)")
                else:
                    logger.warning(f"  Hazard '{h}': NOT ready")

        mem_after = get_current_rss_mb()
        logger.info(
            f"All active models loaded successfully. "
            f"Memory RSS: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB). "
            f"Well within {settings.MAX_MEMORY_RSS_MB} MB free tier limit."
        )
    except Exception as e:
        logger.critical(f"FATAL STARTUP ERROR: Failed to load models: {e}", exc_info=True)
        raise RuntimeError(f"Service startup failed: {e}") from e

    yield

    logger.info("Shutting down AAPADAMITRA Inference Service...")


# FastAPI Application instance
app = FastAPI(
    title="AAPADAMITRA Multi-Hazard AI Platform API",
    description=(
        "Multi-hazard disaster early-warning and decision-support inference service. "
        "Hosts the production-promoted 24-hour Cyclone Track & Intensity model (v3.0) "
        "and demonstration endpoints for Flood, Landslide, and Tsunami hazard screening. "
        "Operates under ₹0 budget on Render Free Web Service with direct local repository model checkout."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# EXCEPTION HANDLERS
# ------------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for pydantic input validation errors."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field,
            "message": err.get("msg"),
            "type": err.get("type")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "status": "VALIDATION_ERROR",
            "message": "Invalid input features provided for disaster inference.",
            "details": errors
        }
    )


# ------------------------------------------------------------------------------
# HEALTH & METADATA ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System Health"])
def health_check(response: Response):
    """
    Health check endpoint reporting server status, uptime, and memory consumption.
    Verifies that the process remains well below Render's 512 MB free tier ceiling.
    """
    try:
        cyc_ready = model_loader.get_cyclone_bundle().is_ready
    except Exception:
        cyc_ready = False

    try:
        flood_ready = model_loader.get_flood_bundle().is_ready
    except Exception:
        flood_ready = False

    try:
        ls_ready = model_loader.get_landslide_bundle().is_ready
    except Exception:
        ls_ready = False

    try:
        ts_ready = model_loader.get_tsunami_bundle().is_ready
    except Exception:
        ts_ready = False

    rss_mb = get_current_rss_mb()
    uptime = round(time.time() - SERVER_START_TIME, 1)
    limit_mb = settings.MAX_MEMORY_RSS_MB
    usage_pct = round((rss_mb / limit_mb) * 100.0, 1)

    is_healthy = cyc_ready and (rss_mb < limit_mb)
    status_str = "HEALTHY" if is_healthy else "UNHEALTHY"

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=status_str,
        service="AAPADAMITRA Multi-Hazard Inference API",
        environment=settings.ENVIRONMENT,
        uptime_seconds=uptime,
        memory_rss_mb=rss_mb,
        memory_limit_mb=limit_mb,
        memory_usage_pct=usage_pct,
        cyclone_model_loaded=cyc_ready,
        flood_model_loaded=flood_ready,
        landslide_model_loaded=ls_ready,
        tsunami_model_loaded=ts_ready,
        demo_mode=settings.DEMO_MODE,
        model_version="v3.0",
        active_hazard="cyclone",
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


@app.get("/models", response_model=ModelsStatusResponse, tags=["Model Registry"])
def list_models_status():
    """
    Lists deployment status for all four hazard modules.
    Explicitly reflects Phase 3 scientific audit decisions:
    - Cyclone: ACTIVE (Promoted to Production)
    - Flood: NOT PROMOTED (Candidate NSE 0.9189 vs Persistence 0.9229; Demonstration Only)
    - Landslide: NOT PROMOTED (Synthetic background separation artifact; Demonstration Only)
    - Tsunami: NOT PROMOTED (Candidate PR-AUC 0.6500 vs Existing 0.6845; Demonstration Only)
    """
    models = {
        "cyclone": HazardStatus(
            hazard="cyclone",
            model_name="cyclone_track_and_intensity_24h",
            version="v3.0",
            production_status="PROMOTED TO PRODUCTION",
            is_active=True,
            demo_enabled=True,
            demo_status="PRODUCTION_APPROVED",
            promotion_rationale="PROMOTED: Track error (136.5km) outperforms operational persistence (161.84km).",
            available_endpoints=["/api/v1/cyclone/predict", "/api/v1/cyclone/model-info"],
            statutory_authority="India Meteorological Department (IMD) - RSMC New Delhi"
        ),
        "flood": HazardStatus(
            hazard="flood",
            model_name="flood_discharge_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="REJECTED: Candidate NSE (0.9189) does not beat operational persistence (0.9229).",
            available_endpoints=["/api/v1/flood/predict", "/api/v1/flood/model-info"],
            statutory_authority="Central Water Commission (CWC) / National Water Informatics Centre (NWIC)"
        ),
        "landslide": HazardStatus(
            hazard="landslide",
            model_name="landslide_susceptibility",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="REJECTED: Candidate PR-AUC (1.0) does not outperform baseline (1.0). Retained in research catalog.",
            available_endpoints=["/api/v1/landslide/predict", "/api/v1/landslide/model-info"],
            statutory_authority="Geological Survey of India (GSI)"
        ),
        "tsunami": HazardStatus(
            hazard="tsunami",
            model_name="tsunami_classifier_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="RETAINED_EXISTING: Candidate PR-AUC (0.65) lower than existing baseline (0.6845).",
            available_endpoints=["/api/v1/tsunami/predict", "/api/v1/tsunami/model-info"],
            statutory_authority="Indian National Centre for Ocean Information Services (INCOIS)"
        )
    }

    return ModelsStatusResponse(
        active_hazard_count=1,
        demonstration_hazard_count=4,
        total_hazards=4,
        models=models
    )


@app.get("/api/v1/hazards", response_model=HazardsListResponse, tags=["Model Registry"])
def list_all_hazards():
    """Returns detailed multi-hazard catalog including demonstration endpoints and authorities."""
    hazards = [
        HazardSummaryItem(
            hazard="cyclone",
            title="Tropical Cyclone Track, Intensity & Rapid Intensification",
            model_name="cyclone_track_and_intensity_24h",
            version="v3.0",
            production_status="PROMOTED TO PRODUCTION",
            demo_enabled=True,
            demo_status="PRODUCTION_APPROVED",
            promotion_rationale="Outperforms operational persistence (136.5 km vs 161.84 km track error).",
            statutory_authority="India Meteorological Department (IMD) - RSMC New Delhi",
            endpoints={
                "predict": "/api/v1/cyclone/predict",
                "model_info": "/api/v1/cyclone/model-info"
            },
            key_metrics={"mean_track_error_km": 136.50, "intensity_mae_kts": 11.30, "ri_pr_auc": 0.2379}
        ),
        HazardSummaryItem(
            hazard="flood",
            title="River Basin Gauging Station Discharge Forecast (+24h)",
            model_name="flood_discharge_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="Candidate NSE (0.9189) did not beat 1-day persistence (0.9229).",
            statutory_authority="Central Water Commission (CWC) / NWIC",
            endpoints={
                "predict": "/api/v1/flood/predict",
                "model_info": "/api/v1/flood/model-info"
            },
            key_metrics={"NSE": 0.9189, "KGE": 0.9394, "persistence_benchmark_NSE": 0.9229}
        ),
        HazardSummaryItem(
            hazard="landslide",
            title="Terrain Morphometry & Hydrometeorological Landslide Susceptibility",
            model_name="landslide_susceptibility",
            version="v3.0",
            production_status="NOT PROMOTED",
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="Candidate PR-AUC (1.0) artifact of synthetic background sampling.",
            statutory_authority="Geological Survey of India (GSI)",
            endpoints={
                "predict": "/api/v1/landslide/predict",
                "model_info": "/api/v1/landslide/model-info"
            },
            key_metrics={"pr_auc": 1.0, "brier_score": 0.0002, "audit_note": "Synthetic background artifact"}
        ),
        HazardSummaryItem(
            hazard="tsunami",
            title="Seismic Hypocenter & Bathymetric Tsunami Potential Early Screening",
            model_name="tsunami_classifier_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            demo_enabled=True,
            demo_status="DEMONSTRATION_ONLY",
            promotion_rationale="Candidate PR-AUC (0.6500) lower than pre-existing baseline (0.6845).",
            statutory_authority="Indian National Centre for Ocean Information Services (INCOIS)",
            endpoints={
                "predict": "/api/v1/tsunami/predict",
                "model_info": "/api/v1/tsunami/model-info"
            },
            key_metrics={"pr_auc": 0.6500, "roc_auc": 0.7708, "pre_existing_pr_auc": 0.6845}
        )
    ]

    return HazardsListResponse(
        total_hazards=4,
        promoted_hazards_count=1,
        demonstration_hazards_count=4,
        operational_mode="MULTI_HAZARD_DEMONSTRATION",
        hazards=hazards,
        system_note=(
            "AAPADAMITRA operates under strict Phase 3 scientific audit governance. Only Cyclone v3.0 "
            "is promoted to production. Flood, Landslide, and Tsunami modules are enabled under Demonstration "
            "Mode with full statutory notices and scientific limitation caveats."
        )
    )


# ------------------------------------------------------------------------------
# CYCLONE INFERENCE ENDPOINTS (PRODUCTION PROMOTED)
# ------------------------------------------------------------------------------
@app.get(
    "/api/v1/cyclone/model-info",
    response_model=ModelInfoResponse,
    tags=["Cyclone Forecasting"]
)
def get_cyclone_model_info():
    """Returns scientific metadata, validation metrics, and limitations for Cyclone v3.0."""
    return cyclone_service.get_model_info()


@app.post(
    "/api/v1/cyclone/predict",
    response_model=CyclonePredictionResponse,
    tags=["Cyclone Forecasting"],
    dependencies=[Depends(verify_api_key)]
)
def predict_cyclone(request: CyclonePredictionRequest):
    """
    Executes 24-hour Tropical Cyclone Track, Intensity & Rapid Intensification Forecast.
    Requires 'X-API-Key' authentication header.
    """
    try:
        return cyclone_service.predict(request)
    except Exception as e:
        logger.error(f"Cyclone prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed. Please verify input parameters and try again."
        )


# ------------------------------------------------------------------------------
# FLOOD INFERENCE ENDPOINTS (DEMONSTRATION MODE)
# ------------------------------------------------------------------------------
@app.get(
    "/api/v1/flood/model-info",
    response_model=FloodModelInfoResponse,
    tags=["Flood Forecasting (Demonstration)"]
)
def get_flood_model_info():
    """Returns scientific metadata and benchmark metrics for candidate Flood discharge model."""
    return flood_service.get_model_info()


@app.post(
    "/api/v1/flood/predict",
    response_model=FloodPredictionResponse,
    tags=["Flood Forecasting (Demonstration)"],
    dependencies=[Depends(verify_api_key)]
)
def predict_flood(request: FloodPredictionRequest):
    """
    Executes 24-hour Gauging Station River Discharge Forecast under Demonstration Mode.
    Compares prediction against 1-day operational persistence benchmark.
    Requires 'X-API-Key' authentication header.
    """
    try:
        return flood_service.predict(request)
    except Exception as e:
        logger.error(f"Flood prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flood inference execution failed. Please verify input parameters."
        )


# ------------------------------------------------------------------------------
# LANDSLIDE INFERENCE ENDPOINTS (DEMONSTRATION MODE)
# ------------------------------------------------------------------------------
@app.get(
    "/api/v1/landslide/model-info",
    response_model=LandslideModelInfoResponse,
    tags=["Landslide Screening (Demonstration)"]
)
def get_landslide_model_info():
    """Returns scientific metadata and audit caveats for candidate Landslide model."""
    return landslide_service.get_model_info()


@app.post(
    "/api/v1/landslide/predict",
    response_model=LandslidePredictionResponse,
    tags=["Landslide Screening (Demonstration)"],
    dependencies=[Depends(verify_api_key)]
)
def predict_landslide(request: LandslidePredictionRequest):
    """
    Executes Landslide Susceptibility Screening under Demonstration Mode.
    Requires 'X-API-Key' authentication header.
    """
    try:
        return landslide_service.predict(request)
    except Exception as e:
        logger.error(f"Landslide prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Landslide inference execution failed. Please verify input parameters."
        )


# ------------------------------------------------------------------------------
# TSUNAMI INFERENCE ENDPOINTS (DEMONSTRATION MODE)
# ------------------------------------------------------------------------------
@app.get(
    "/api/v1/tsunami/model-info",
    response_model=TsunamiModelInfoResponse,
    tags=["Tsunami Screening (Demonstration)"]
)
def get_tsunami_model_info():
    """Returns scientific metadata and benchmark metrics for candidate Tsunami screening model."""
    return tsunami_service.get_model_info()


@app.post(
    "/api/v1/tsunami/predict",
    response_model=TsunamiPredictionResponse,
    tags=["Tsunami Screening (Demonstration)"],
    dependencies=[Depends(verify_api_key)]
)
def predict_tsunami(request: TsunamiPredictionRequest):
    """
    Executes Tsunami Potential Early Screening under Demonstration Mode.
    Evaluates seismic hypocenter parameters against 0.44 F2-tuned validation threshold.
    Requires 'X-API-Key' authentication header.
    """
    try:
        return tsunami_service.predict(request)
    except Exception as e:
        logger.error(f"Tsunami prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tsunami inference execution failed. Please verify input parameters."
        )


# ------------------------------------------------------------------------------
# GUARD FOR UNKNOWN HAZARDS
# ------------------------------------------------------------------------------
@app.api_route(
    "/api/v1/{hazard}/predict",
    methods=["GET", "POST", "PUT", "DELETE"],
    tags=["Hazard Guard"],
    include_in_schema=False
)
def guard_unknown_hazards(hazard: str):
    """Rejects requests for unhandled hazard types."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Disaster hazard module '{hazard}' does not exist in AAPADAMITRA."
    )

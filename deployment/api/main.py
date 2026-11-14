"""
AAPADAMITRA Multi-Hazard AI Platform - Inference API Server
FastAPI application optimized for Render Free Web Service (512 MB RAM limit).
Strictly serves the production-promoted Cyclone v3.0 model while explicitly
guarding unpromoted hazard modules (Flood, Landslide, Tsunami).
Loads model artifacts directly from the repository checkout with zero cloud storage dependencies.
"""

from typing import Dict, Any
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
    CyclonePredictionRequest,
    CyclonePredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ModelsStatusResponse,
    HazardStatus
)
from deployment.api.security import verify_api_key
from deployment.api.model_loader import model_loader
from deployment.api.cyclone_service import cyclone_service

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
        # Load Cyclone models directly from local repository
        bundle = model_loader.load_cyclone_models()
        mem_after = get_current_rss_mb()
        logger.info(
            f"Cyclone v3.0 models loaded successfully in {bundle.load_time_seconds:.2f}s. "
            f"Memory RSS after loading: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB). "
            f"Well within {settings.MAX_MEMORY_RSS_MB} MB free tier limit."
        )
    except Exception as e:
        logger.critical(f"FATAL STARTUP ERROR: Failed to load approved Cyclone models: {e}", exc_info=True)
        raise RuntimeError(f"Service startup failed: {e}") from e

    yield

    logger.info("Shutting down AAPADAMITRA Inference Service...")


# FastAPI Application instance
app = FastAPI(
    title="AAPADAMITRA Multi-Hazard AI Platform API",
    description=(
        "Production inference service for AAPADAMITRA. "
        "Hosts the promoted 24-hour Cyclone Track, Intensity & Rapid Intensification model (v3.0). "
        "Operates under strictly ₹0 budget on Render Free Web Service with direct local repository model checkout."
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
    Returns HTTP 503 if models are unavailable.
    """
    try:
        bundle = model_loader.get_bundle()
        is_ready = bundle.is_ready
    except Exception as e:
        logger.error(f"Health check failed to access model bundle: {e}")
        is_ready = False

    rss_mb = get_current_rss_mb()
    uptime = round(time.time() - SERVER_START_TIME, 1)
    limit_mb = settings.MAX_MEMORY_RSS_MB
    usage_pct = round((rss_mb / limit_mb) * 100.0, 1)

    is_healthy = is_ready and (rss_mb < limit_mb)
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
        cyclone_model_loaded=is_ready,
        model_version="v3.0",
        active_hazard="cyclone",
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


@app.get("/models", response_model=ModelsStatusResponse, tags=["Model Registry"])
def list_models_status():
    """
    Lists deployment status for all four hazard modules.
    Explicitly reflects Phase 3 scientific audit decisions:
    - Cyclone: ACTIVE / PROMOTED
    - Flood: NOT PROMOTED (Candidate did not beat persistence)
    - Landslide: NOT PROMOTED (Synthetic separation & spatial leakage)
    - Tsunami: NOT PROMOTED (Candidate lower than existing baseline)
    """
    models = {
        "cyclone": HazardStatus(
            hazard="cyclone",
            model_name="cyclone_track_and_intensity_24h",
            version="v3.0",
            production_status="PROMOTED TO PRODUCTION",
            is_active=True,
            promotion_rationale="PROMOTED: Track error (136.5km) outperforms operational persistence (161.84km).",
            available_endpoints=["/api/v1/cyclone/predict", "/api/v1/cyclone/model-info"]
        ),
        "flood": HazardStatus(
            hazard="flood",
            model_name="flood_discharge_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            promotion_rationale="REJECTED: Candidate NSE (0.9189) does not beat operational persistence (0.9229).",
            available_endpoints=[]
        ),
        "landslide": HazardStatus(
            hazard="landslide",
            model_name="landslide_susceptibility",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            promotion_rationale="REJECTED: Candidate PR-AUC (1.0) does not outperform baseline (1.0). Retained in research catalog.",
            available_endpoints=[]
        ),
        "tsunami": HazardStatus(
            hazard="tsunami",
            model_name="tsunami_classifier_rf",
            version="v3.0",
            production_status="NOT PROMOTED",
            is_active=False,
            promotion_rationale="RETAINED_EXISTING: Candidate PR-AUC (0.65) lower than existing baseline (0.6845).",
            available_endpoints=[]
        )
    }

    return ModelsStatusResponse(
        active_hazard_count=1,
        total_hazards=4,
        models=models
    )


# ------------------------------------------------------------------------------
# CYCLONE INFERENCE ENDPOINTS
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
        logger.error(f"Prediction inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed. Please verify input parameters and try again."
        )


# ------------------------------------------------------------------------------
# UNPROMOTED HAZARD ENDPOINT GUARDS
# ------------------------------------------------------------------------------
@app.api_route(
    "/api/v1/{hazard}/predict",
    methods=["GET", "POST", "PUT", "DELETE"],
    tags=["Unpromoted Hazard Guards"],
    include_in_schema=False
)
def guard_unpromoted_hazards(hazard: str):
    """
    Explicit guard rejecting inference requests for unpromoted disaster modules.
    Guarantees that unapproved models (Flood, Landslide, Tsunami) cannot be invoked.
    """
    hazard_lower = hazard.lower()
    if hazard_lower in ["flood", "landslide", "tsunami"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Hazard module '{hazard_lower}' is NOT PROMOTED to production. "
                "Under AAPADAMITRA Phase 3 scientific audit criteria, only the Cyclone model "
                "meets production standards. Check GET /models for details."
            )
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Disaster hazard module '{hazard}' does not exist."
    )

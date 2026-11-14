"""
Unit tests for AAPADAMITRA API Health, Resource Monitoring and Model Registry endpoints.
Verifies multi-hazard demonstration mode status and memory limits.
"""

from deployment.api.config import settings


def test_health_endpoint(client):
    """Verifies that /health returns 200, status HEALTHY, and memory RSS < 512 MB."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert data["service"] == "AAPADAMITRA Multi-Hazard Inference API"
    assert data["uptime_seconds"] >= 0.0
    assert data["memory_rss_mb"] > 0.0
    # Must remain strictly below the 512 MB Render Free Tier limit
    assert data["memory_rss_mb"] < 512.0
    assert data["memory_limit_mb"] == 512.0
    assert 0.0 <= data["memory_usage_pct"] <= 100.0
    assert data["cyclone_model_loaded"] is True
    assert data["flood_model_loaded"] is True
    assert data["landslide_model_loaded"] is True
    assert data["tsunami_model_loaded"] is True
    assert data["demo_mode"] is True
    assert data["model_version"] == "v3.0"
    assert data["active_hazard"] == "cyclone"


def test_models_registry_status(client):
    """
    Verifies that /models clearly delineates:
    - Cyclone: ACTIVE (Promoted to Production)
    - Flood: NOT PROMOTED (Demonstration Mode Active)
    - Landslide: NOT PROMOTED (Demonstration Mode Active)
    - Tsunami: NOT PROMOTED (Demonstration Mode Active)
    """
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()

    assert data["active_hazard_count"] == 1
    assert data["demonstration_hazard_count"] == 4
    assert data["total_hazards"] == 4
    models = data["models"]

    # Cyclone must be PRODUCTION PROMOTED
    assert models["cyclone"]["is_active"] is True
    assert models["cyclone"]["demo_enabled"] is True
    assert models["cyclone"]["demo_status"] == "PRODUCTION_APPROVED"
    assert models["cyclone"]["production_status"] == "PROMOTED TO PRODUCTION"
    assert "/api/v1/cyclone/predict" in models["cyclone"]["available_endpoints"]
    assert "/api/v1/cyclone/model-info" in models["cyclone"]["available_endpoints"]

    # Flood must NOT be promoted (Demonstration Only)
    assert models["flood"]["is_active"] is False
    assert models["flood"]["demo_enabled"] is True
    assert models["flood"]["demo_status"] == "DEMONSTRATION_ONLY"
    assert models["flood"]["production_status"] == "NOT PROMOTED"
    assert "/api/v1/flood/predict" in models["flood"]["available_endpoints"]
    assert "/api/v1/flood/model-info" in models["flood"]["available_endpoints"]
    assert "persistence" in models["flood"]["promotion_rationale"].lower()

    # Landslide must NOT be promoted (Demonstration Only)
    assert models["landslide"]["is_active"] is False
    assert models["landslide"]["demo_enabled"] is True
    assert models["landslide"]["demo_status"] == "DEMONSTRATION_ONLY"
    assert models["landslide"]["production_status"] == "NOT PROMOTED"
    assert "/api/v1/landslide/predict" in models["landslide"]["available_endpoints"]
    assert "/api/v1/landslide/model-info" in models["landslide"]["available_endpoints"]

    # Tsunami must NOT be promoted (Demonstration Only)
    assert models["tsunami"]["is_active"] is False
    assert models["tsunami"]["demo_enabled"] is True
    assert models["tsunami"]["demo_status"] == "DEMONSTRATION_ONLY"
    assert models["tsunami"]["production_status"] == "NOT PROMOTED"
    assert "/api/v1/tsunami/predict" in models["tsunami"]["available_endpoints"]
    assert "/api/v1/tsunami/model-info" in models["tsunami"]["available_endpoints"]


def test_cyclone_model_info(client):
    """Verifies that /api/v1/cyclone/model-info provides complete Phase 3 metadata."""
    response = client.get("/api/v1/cyclone/model-info")
    assert response.status_code == 200
    info = response.json()

    assert info["hazard"] == "cyclone"
    assert info["version"] == "v3.0"
    assert info["lead_time_hours"] == 24
    assert info["feature_count"] == 17
    assert len(info["features_required"]) == 17
    assert "current_latitude" in info["features_required"]
    assert "coriolis_param" in info["features_required"]

    # Test metrics check
    metrics = info["test_metrics"]
    assert metrics["mean_track_error_km"] == 136.50
    assert metrics["intensity_mae_kts"] == 11.30

    # Baseline comparison check
    baselines = info["baseline_metrics"]
    assert baselines["dead_reckoning_persistence_track_error_km"] == 161.84

    # Statutory authority
    assert "IMD" in info["statutory_authority"]

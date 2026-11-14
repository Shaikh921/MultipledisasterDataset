"""
Unit tests for API Key Authentication on secured endpoints across all four hazard modules.
Enforces that requests require authentication and that default dev keys are blocked in production.
"""

from unittest.mock import patch
from deployment.api.config import settings

SAMPLE_CYCLONE_PAYLOAD = {
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
    "storm_name": "REMAL"
}

SAMPLE_FLOOD_PAYLOAD = {
    "station_name": "Baltara",
    "river_name": "Kosi",
    "q_t": 1250.0,
    "q_lag1": 1180.0,
    "q_lag2": 1100.0,
    "q_lag3": 1050.0
}

SAMPLE_LANDSLIDE_PAYLOAD = {
    "elevation_dem_m": 1240.0,
    "slope_degrees": 34.5,
    "aspect_degrees": 180.0,
    "profile_curvature": -0.05,
    "topographic_wetness_index": 6.8,
    "antecedent_rainfall_3d_mm": 135.0,
    "antecedent_rainfall_7d_mm": 240.0,
    "soil_moisture_0_7cm_m3m3": 0.44,
    "soil_moisture_rootzone_m3m3": 0.40
}

SAMPLE_TSUNAMI_PAYLOAD = {
    "magnitude": 8.2,
    "focal_depth_km": 15.0,
    "epicenter_latitude": 3.3,
    "epicenter_longitude": 95.8,
    "gebco_bathymetry_m": -3500.0,
    "distance_to_nearest_coast_km": 45.0
}


def test_unauthenticated_predict_requests(client):
    """All 4 hazard predict endpoints must return 401 when no X-API-Key header is provided."""
    endpoints = [
        ("/api/v1/cyclone/predict", SAMPLE_CYCLONE_PAYLOAD),
        ("/api/v1/flood/predict", SAMPLE_FLOOD_PAYLOAD),
        ("/api/v1/landslide/predict", SAMPLE_LANDSLIDE_PAYLOAD),
        ("/api/v1/tsunami/predict", SAMPLE_TSUNAMI_PAYLOAD),
    ]
    for path, payload in endpoints:
        response = client.post(path, json=payload)
        assert response.status_code == 401, f"{path} did not reject unauthenticated request"
        assert "Missing API key" in response.json()["detail"]


def test_invalid_api_key_predict_request(client):
    """Predict endpoints must return 401 when an invalid X-API-Key is supplied."""
    headers = {"X-API-Key": "wrong-secret-token"}
    response = client.post("/api/v1/cyclone/predict", json=SAMPLE_CYCLONE_PAYLOAD, headers=headers)
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_valid_api_key_predict_requests(client):
    """All 4 hazard predict endpoints must succeed (200 OK) when valid X-API-Key is provided."""
    headers = {"X-API-Key": settings.API_KEY}
    endpoints = [
        ("/api/v1/cyclone/predict", SAMPLE_CYCLONE_PAYLOAD),
        ("/api/v1/flood/predict", SAMPLE_FLOOD_PAYLOAD),
        ("/api/v1/landslide/predict", SAMPLE_LANDSLIDE_PAYLOAD),
        ("/api/v1/tsunami/predict", SAMPLE_TSUNAMI_PAYLOAD),
    ]
    for path, payload in endpoints:
        response = client.post(path, json=payload, headers=headers)
        assert response.status_code == 200, f"{path} failed with valid key: {response.text}"
        assert response.json()["status"] == "SUCCESS"


def test_production_mode_rejects_default_dev_key(client):
    """In production mode, the server must refuse to authorize requests with dev keys."""
    with patch.object(settings, "ENVIRONMENT", "production"):
        headers = {"X-API-Key": "aapadamitra-dev-key"}
        response = client.post("/api/v1/cyclone/predict", json=SAMPLE_CYCLONE_PAYLOAD, headers=headers)
        assert response.status_code == 500
        assert "production api key" in response.json()["detail"].lower()


def test_public_endpoints_accessible_without_auth(client):
    """Health, registry status, docs, hazards catalog and model-info must be publicly accessible."""
    public_paths = [
        "/health",
        "/models",
        "/api/v1/hazards",
        "/api/v1/cyclone/model-info",
        "/api/v1/flood/model-info",
        "/api/v1/landslide/model-info",
        "/api/v1/tsunami/model-info",
        "/docs",
        "/openapi.json"
    ]
    for path in public_paths:
        res = client.get(path)
        assert res.status_code == 200, f"Endpoint {path} unexpectedly required auth"

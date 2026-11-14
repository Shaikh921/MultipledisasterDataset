"""
Unit tests for API Key Authentication on secured endpoints.
Enforces that requests require authentication and that default dev keys are blocked in production.
"""

from unittest.mock import patch
from deployment.api.config import settings

SAMPLE_PAYLOAD = {
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


def test_unauthenticated_predict_request(client):
    """Predict endpoint must return 401 when no X-API-Key header is provided."""
    response = client.post("/api/v1/cyclone/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


def test_invalid_api_key_predict_request(client):
    """Predict endpoint must return 401 when an invalid X-API-Key is supplied."""
    headers = {"X-API-Key": "wrong-secret-token"}
    response = client.post("/api/v1/cyclone/predict", json=SAMPLE_PAYLOAD, headers=headers)
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_valid_api_key_predict_request(client):
    """Predict endpoint must succeed (200 OK) when valid X-API-Key is provided."""
    headers = {"X-API-Key": settings.API_KEY}
    response = client.post("/api/v1/cyclone/predict", json=SAMPLE_PAYLOAD, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"


def test_production_mode_rejects_default_dev_key(client):
    """In production mode, the server must refuse to authorize requests with dev keys."""
    with patch.object(settings, "ENVIRONMENT", "production"):
        headers = {"X-API-Key": "aapadamitra-dev-key"}
        response = client.post("/api/v1/cyclone/predict", json=SAMPLE_PAYLOAD, headers=headers)
        assert response.status_code == 500
        assert "production api key" in response.json()["detail"].lower()


def test_public_endpoints_accessible_without_auth(client):
    """Health, registry status, docs and model info must be publicly accessible."""
    for path in ["/health", "/models", "/api/v1/cyclone/model-info", "/docs", "/openapi.json"]:
        res = client.get(path)
        assert res.status_code == 200, f"Endpoint {path} unexpectedly required auth"

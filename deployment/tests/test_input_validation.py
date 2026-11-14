"""
Unit tests for Pydantic input validation and unpromoted hazard guards.
"""

from deployment.api.config import settings

AUTH_HEADERS = {"X-API-Key": settings.API_KEY}


def test_missing_required_fields(client):
    """Omitting mandatory meteorological features must return 422."""
    incomplete_payload = {
        "current_latitude": 15.0,
        "current_longitude": 85.0
        # missing wind speed, pressure, shear, etc.
    }
    response = client.post("/api/v1/cyclone/predict", json=incomplete_payload, headers=AUTH_HEADERS)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "VALIDATION_ERROR"
    assert len(data["details"]) > 0


def test_latitude_out_of_bounds(client):
    """Latitude > 90 or < -90 must fail validation with 422."""
    payload = {
        "current_latitude": 95.0,  # Invalid
        "current_longitude": 89.0,
        "current_wind_speed_kts_3min": 45.0,
        "current_central_pressure_hpa": 980.0,
        "dist2land_km": 200.0,
        "translation_speed_kts": 12.0,
        "heading_degrees": 330.0,
        "sea_surface_temperature_c": 29.0,
        "vertical_wind_shear_kts": 12.0,
        "relative_humidity_700hpa_pct": 75.0
    }
    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_longitude_out_of_bounds(client):
    """Longitude > 180 or < -180 must fail validation with 422."""
    payload = {
        "current_latitude": 18.0,
        "current_longitude": 210.0,  # Invalid
        "current_wind_speed_kts_3min": 45.0,
        "current_central_pressure_hpa": 980.0,
        "dist2land_km": 200.0,
        "translation_speed_kts": 12.0,
        "heading_degrees": 330.0,
        "sea_surface_temperature_c": 29.0,
        "vertical_wind_shear_kts": 12.0,
        "relative_humidity_700hpa_pct": 75.0
    }
    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_heading_out_of_bounds(client):
    """Heading outside [0, 360] must fail validation with 422."""
    payload = {
        "current_latitude": 18.0,
        "current_longitude": 88.0,
        "current_wind_speed_kts_3min": 45.0,
        "current_central_pressure_hpa": 980.0,
        "dist2land_km": 200.0,
        "translation_speed_kts": 12.0,
        "heading_degrees": 400.0,  # Invalid
        "sea_surface_temperature_c": 29.0,
        "vertical_wind_shear_kts": 12.0,
        "relative_humidity_700hpa_pct": 75.0
    }
    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_relative_humidity_out_of_bounds(client):
    """RH > 100% or < 0% must fail validation with 422."""
    payload = {
        "current_latitude": 18.0,
        "current_longitude": 88.0,
        "current_wind_speed_kts_3min": 45.0,
        "current_central_pressure_hpa": 980.0,
        "dist2land_km": 200.0,
        "translation_speed_kts": 12.0,
        "heading_degrees": 300.0,
        "sea_surface_temperature_c": 29.0,
        "vertical_wind_shear_kts": 12.0,
        "relative_humidity_700hpa_pct": 110.0  # Invalid
    }
    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_demonstration_hazards_are_accessible_and_validated(client):
    """
    Flood, Landslide, and Tsunami endpoints are accessible in Demonstration Mode.
    Passing empty payload `{}` triggers 422 Unprocessable Content validation.
    """
    for demo_hazard in ["flood", "landslide", "tsunami"]:
        response = client.post(f"/api/v1/{demo_hazard}/predict", json={}, headers=AUTH_HEADERS)
        assert response.status_code == 422, f"Expected 422 validation for {demo_hazard}"
        assert response.json()["status"] == "VALIDATION_ERROR"


def test_unknown_hazard_returns_404(client):
    """Requesting an unrecognized hazard endpoint must return 404 Not Found."""
    for unknown in ["volcano", "wildfire", "earthquake", "tornado"]:
        response = client.post(f"/api/v1/{unknown}/predict", json={}, headers=AUTH_HEADERS)
        assert response.status_code == 404

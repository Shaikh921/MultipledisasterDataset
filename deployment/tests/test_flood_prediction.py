"""
Unit tests for Flood River Discharge Prediction & Model Info endpoints.
"""

from deployment.api.config import settings

SAMPLE_FLOOD_PAYLOAD = {
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


def test_flood_prediction_success(client):
    """Verifies that POST /api/v1/flood/predict executes inference and returns structured response."""
    headers = {"X-API-Key": settings.API_KEY}
    response = client.post("/api/v1/flood/predict", json=SAMPLE_FLOOD_PAYLOAD, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["execution_mode"] == "DEMONSTRATION"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["forecast_horizon_hours"] == 24
    assert data["station_name"] == "Baltara"

    # Prediction outcomes
    preds = data["predictions"]
    assert preds["predicted_discharge_m3_s"] >= 0.0
    assert preds["persistence_discharge_m3_s"] == 1250.0
    assert "estimated_flood_severity" in preds

    # Scientific and statutory notices
    assert "CWC" in data["statutory_disclaimer"]
    assert "persistence" in data["scientific_caveat"].lower()


def test_flood_prediction_missing_feature(client):
    """Verifies that omitting a required discharge feature (q_t) returns 422 Validation Error."""
    headers = {"X-API-Key": settings.API_KEY}
    bad_payload = dict(SAMPLE_FLOOD_PAYLOAD)
    del bad_payload["q_t"]

    response = client.post("/api/v1/flood/predict", json=bad_payload, headers=headers)
    assert response.status_code == 422
    assert response.json()["status"] == "VALIDATION_ERROR"


def test_flood_model_info(client):
    """Verifies that GET /api/v1/flood/model-info returns metadata and persistence benchmark."""
    response = client.get("/api/v1/flood/model-info")
    assert response.status_code == 200
    data = response.json()

    assert data["hazard"] == "flood"
    assert data["version"] == "v3.0"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["test_metrics"]["NSE"] == 0.9189
    assert data["baseline_metrics"]["1_day_persistence_NSE"] == 0.9229
    assert "CWC" in data["statutory_authority"]


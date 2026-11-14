"""
Unit tests for Tsunami Potential Early Screening & Model Info endpoints.
"""

from deployment.api.config import settings

SAMPLE_TSUNAMI_PAYLOAD = {
    "event_id": "usgs_megathrust_analogue",
    "magnitude": 8.2,
    "focal_depth_km": 15.0,
    "epicenter_latitude": 3.3,
    "epicenter_longitude": 95.8,
    "gebco_bathymetry_m": -3500.0,
    "distance_to_nearest_coast_km": 45.0
}


def test_tsunami_prediction_success(client):
    """Verifies that POST /api/v1/tsunami/predict executes early screening inference."""
    headers = {"X-API-Key": settings.API_KEY}
    response = client.post("/api/v1/tsunami/predict", json=SAMPLE_TSUNAMI_PAYLOAD, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["execution_mode"] == "DEMONSTRATION"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["event_id"] == "usgs_megathrust_analogue"

    preds = data["predictions"]
    assert 0.0 <= preds["tsunamigenic_probability"] <= 1.0
    assert preds["decision_threshold"] == 0.44
    assert preds["tsunami_screening_alert"] in [True, False]
    assert preds["empirical_rule_triggered"] is True  # Mw 8.2, depth 15km, bathy -3500m triggers rule

    # Scientific and statutory notices
    assert "INCOIS" in data["statutory_disclaimer"]
    assert "hypocentral" in data["scientific_caveat"].lower()


def test_tsunami_prediction_invalid_magnitude(client):
    """Verifies that invalid magnitude (>10.0) returns 422 Validation Error."""
    headers = {"X-API-Key": settings.API_KEY}
    bad_payload = dict(SAMPLE_TSUNAMI_PAYLOAD)
    bad_payload["magnitude"] = 12.0  # Invalid Mw > 10.0

    response = client.post("/api/v1/tsunami/predict", json=bad_payload, headers=headers)
    assert response.status_code == 422
    assert response.json()["status"] == "VALIDATION_ERROR"


def test_tsunami_model_info(client):
    """Verifies that GET /api/v1/tsunami/model-info returns metadata and validation threshold."""
    response = client.get("/api/v1/tsunami/model-info")
    assert response.status_code == 200
    data = response.json()

    assert data["hazard"] == "tsunami"
    assert data["version"] == "v3.0"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["test_metrics"]["pr_auc"] == 0.6500
    assert data["test_metrics"]["validation_tuned_threshold"] == 0.44
    assert "INCOIS" in data["statutory_authority"]


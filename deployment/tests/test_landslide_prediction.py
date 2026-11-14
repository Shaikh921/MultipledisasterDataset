"""
Unit tests for Landslide Susceptibility Prediction & Model Info endpoints.
"""

from deployment.api.config import settings

SAMPLE_LANDSLIDE_PAYLOAD = {
    "location_name": "Wayanad Hills",
    "elevation_dem_m": 1240.0,
    "slope_degrees": 34.5,
    "aspect_degrees": 180.0,
    "profile_curvature": -0.05,
    "topographic_wetness_index": 6.8,
    "soil_thickness_m": 1.5,
    "antecedent_rainfall_3d_mm": 135.0,
    "antecedent_rainfall_7d_mm": 240.0,
    "soil_moisture_0_7cm_m3m3": 0.44,
    "soil_moisture_rootzone_m3m3": 0.40
}


def test_landslide_prediction_success(client):
    """Verifies that POST /api/v1/landslide/predict executes inference with calibrated probabilities."""
    headers = {"X-API-Key": settings.API_KEY}
    response = client.post("/api/v1/landslide/predict", json=SAMPLE_LANDSLIDE_PAYLOAD, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["execution_mode"] == "DEMONSTRATION"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["location_name"] == "Wayanad Hills"

    preds = data["predictions"]
    assert 0.0 <= preds["susceptibility_probability"] <= 1.0
    assert preds["high_risk_alert"] in [True, False]
    assert len(preds["critical_trigger_factors"]) > 0

    # Scientific and statutory notices
    assert "GSI" in data["statutory_disclaimer"]
    assert "synthetic background" in data["scientific_caveat"].lower()


def test_landslide_prediction_out_of_bounds(client):
    """Verifies that an out-of-bounds slope (>90 degrees) returns 422 Validation Error."""
    headers = {"X-API-Key": settings.API_KEY}
    bad_payload = dict(SAMPLE_LANDSLIDE_PAYLOAD)
    bad_payload["slope_degrees"] = 120.0  # Invalid slope > 90

    response = client.post("/api/v1/landslide/predict", json=bad_payload, headers=headers)
    assert response.status_code == 422
    assert response.json()["status"] == "VALIDATION_ERROR"


def test_landslide_model_info(client):
    """Verifies that GET /api/v1/landslide/model-info returns metadata and audit caveats."""
    response = client.get("/api/v1/landslide/model-info")
    assert response.status_code == 200
    data = response.json()

    assert data["hazard"] == "landslide"
    assert data["version"] == "v3.0"
    assert data["production_status"] == "NOT PROMOTED"
    assert data["demonstration_status"] == "DEMONSTRATION_ACTIVE"
    assert data["test_metrics"]["pr_auc"] == 1.0
    assert "GSI" in data["statutory_authority"]


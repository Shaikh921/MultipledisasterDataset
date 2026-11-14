"""
Unit and integration tests for Cyclone 24-hour Inference Engine.
Validates real LightGBM predictions against actual storm fixes (e.g. Cyclone Remal 2024),
Rapid Intensification alerting, IMD intensity mapping, and statutory disclaimers.
"""

from deployment.api.config import settings
from deployment.api.cyclone_service import classify_imd_intensity

AUTH_HEADERS = {"X-API-Key": settings.API_KEY}


def test_remal_2024_prediction(client):
    """
    Tests end-to-end 24-hour prediction on real May 25, 2024 fix of Cyclone Remal.
    Fix observed: 18.8°N, 89.4°E; actual 24h verify: 20.9°N, 88.7°E, 50 kts.
    """
    payload = {
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

    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    res = response.json()

    # Meta checks
    assert res["status"] == "SUCCESS"
    assert res["model_version"] == "v3.0"
    assert res["forecast_horizon_hours"] == 24
    assert res["storm_name"] == "REMAL"

    # Position checks (should accurately track northward into head of Bay of Bengal)
    preds = res["predictions"]
    assert 19.5 <= preds["predicted_latitude"] <= 21.5
    assert 87.5 <= preds["predicted_longitude"] <= 89.5
    assert preds["delta_latitude"] > 0.0  # Moving North

    # Intensity checks
    assert preds["predicted_wind_speed_kts"] >= 15.0
    assert preds["imd_intensity_category"] == "Cyclonic Storm (34-47 kts)"

    # Rapid Intensification checks
    assert 0.0 <= preds["rapid_intensification_probability"] <= 1.0
    # Remal at this stage had shear = 16.2 kts (> 15 kts threshold), so RI was not triggered
    assert preds["rapid_intensification_alert"] is False

    # Reference Performance (Historical Context, not fabricated individual confidence)
    ref = res["reference_performance"]
    assert ref["mean_track_error_km"] == 136.50
    assert ref["intensity_mae_kts"] == 11.30
    assert ref["baseline_persistence_track_error_km"] == 161.84

    # Statutory Demarcation
    assert "IMD" in res["statutory_disclaimer"]
    assert "New Delhi" in res["statutory_disclaimer"]
    assert "RSMC" in res["statutory_disclaimer"]


def test_rapid_intensification_high_potential_case(client):
    """
    Tests prediction on a storm situated in highly favorable oceanic and atmospheric conditions:
    Very warm SST (31.0°C), very weak vertical shear (5 kts), high mid-level humidity (90%).
    """
    payload = {
        "current_latitude": 12.0,
        "current_longitude": 87.0,
        "current_wind_speed_kts_3min": 45.0,
        "current_central_pressure_hpa": 985.0,
        "dist2land_km": 600.0,
        "translation_speed_kts": 8.0,
        "heading_degrees": 340.0,
        "sea_surface_temperature_c": 31.0,
        "vertical_wind_shear_kts": 5.0,  # Favorable shear
        "relative_humidity_700hpa_pct": 90.0,  # Moist mid-troposphere
        "storm_name": "SYNTH_RI_TEST"
    }

    response = client.post("/api/v1/cyclone/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    res = response.json()
    preds = res["predictions"]

    assert 0.0 <= preds["rapid_intensification_probability"] <= 1.0
    assert preds["rapid_intensification_alert"] == (preds["rapid_intensification_probability"] >= 0.35)


def test_imd_intensity_classification_stages():
    """Verifies that sustained 3-minute wind speeds map to exact statutory IMD categories."""
    assert "Low Pressure Area" in classify_imd_intensity(15.0)
    assert "Depression" in classify_imd_intensity(22.0)
    assert "Deep Depression" in classify_imd_intensity(30.0)
    assert "Cyclonic Storm" in classify_imd_intensity(42.0)
    assert "Severe Cyclonic Storm" in classify_imd_intensity(55.0)
    assert "Very Severe Cyclonic Storm" in classify_imd_intensity(75.0)
    assert "Extremely Severe Cyclonic Storm" in classify_imd_intensity(100.0)
    assert "Super Cyclonic Storm" in classify_imd_intensity(135.0)

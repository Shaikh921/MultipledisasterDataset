"""
AAPADAMITRA Remote API - Example Programmatic Usage
Demonstrates how teammates can import AapadamitraClient in their Python scripts
and run predictions on live or historical tropical cyclone fixes.
"""

import json
from deployment.client.remote_client import AapadamitraClient


def main():
    print("=" * 70)
    print("AAPADAMITRA Client SDK - Example Usage Demonstration")
    print("=" * 70)

    # Initialize client (points to local dev server or Render deployment)
    client = AapadamitraClient(
        base_url="http://localhost:8000",
        api_key="aapadamitra-dev-key"
    )

    # 1. Health and Memory Footprint Check
    print("\n1. Querying API Health & Resource Consumption...")
    try:
        health = client.health_check()
        print(f"   Status:       {health['status']}")
        print(f"   Memory RSS:   {health['memory_rss_mb']:.1f} MB / {health['memory_limit_mb']:.0f} MB ({health['memory_usage_pct']:.1f}%)")
        print(f"   Model Loaded: {health['cyclone_model_loaded']} (Version: {health['model_version']})")
    except Exception as e:
        print(f"   Health check failed (is the server running?): {e}")
        return

    # 2. Multi-Hazard Deployment Status
    print("\n2. Checking Multi-Hazard Registry Status...")
    models_status = client.list_models()
    for hazard, info in models_status["models"].items():
        status_tag = "ACTIVE (Production)" if info["is_active"] else "NOT PROMOTED (Research Catalog)"
        print(f"   - {hazard.upper():<10}: {status_tag}")

    # 3. Model Information & Validation Metrics
    print("\n3. Fetching Cyclone Model Information...")
    info = client.get_cyclone_model_info()
    print(f"   Model Name:          {info['model_name']} ({info['version']})")
    print(f"   Architecture:        {info['architecture']}")
    print(f"   Mean 24h Track Error:{info['test_metrics']['mean_track_error_km']:.2f} km (Holdout 2021-2024)")
    print(f"   Persistence Baseline:{info['baseline_metrics']['dead_reckoning_persistence_track_error_km']:.2f} km")

    # 4. Predict on a Real Cyclone Fix (Cyclone Remal - May 25, 2024, Bay of Bengal)
    print("\n4. Running 24-Hour Forecast on Cyclone Remal Fix (2024-05-25 12:00 UTC)...")
    remal_payload = {
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

    forecast = client.predict_cyclone(remal_payload)
    preds = forecast["predictions"]

    print("\n   --- Forecast Results (+24h Lead) ---")
    print(f"   Predicted Position:  ({preds['predicted_latitude']:.2f}°N, {preds['predicted_longitude']:.2f}°E)")
    print(f"   Actual Verification: (20.90°N, 88.70°E)")
    print(f"   Forecast Wind Speed: {preds['predicted_wind_speed_kts']:.1f} kts (Delta: {preds['delta_wind_speed_kts']:+.1f} kts)")
    print(f"   Actual Wind Speed:   50.0 kts")
    print(f"   IMD Intensity Stage: {preds['imd_intensity_category']}")
    print(f"   Rapid Intensification: {preds['rapid_intensification_probability'] * 100:.1f}% (Alert: {preds['rapid_intensification_alert']})")
    print(f"\n   Statutory Disclaimer:")
    print(f"   {forecast['statutory_disclaimer'][:140]}...")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

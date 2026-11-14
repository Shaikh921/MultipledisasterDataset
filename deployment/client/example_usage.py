"""
AAPADAMITRA Remote API - Multi-Hazard Programmatic Usage Demonstration
Demonstrates how teammates can import AapadamitraClient in their Python scripts
and run predictions across all four natural disaster modules:
1. Cyclone (Production Promoted)
2. Flood (Demonstration Mode)
3. Landslide (Demonstration Mode)
4. Tsunami (Demonstration Mode)
"""

import json
from deployment.client.remote_client import AapadamitraClient


def main():
    print("=" * 75)
    print("AAPADAMITRA Multi-Hazard AI Platform - Client SDK Demonstration")
    print("=" * 75)

    client = AapadamitraClient(
        base_url="http://localhost:8000",
        api_key="aapadamitra-dev-key"
    )

    # 1. Health and Memory Footprint Check
    print("\n[1] Querying Server Health & Resource Footprint...")
    try:
        health = client.health_check()
        print(f"    Status:         {health['status']}")
        print(f"    Memory RSS:     {health['memory_rss_mb']:.1f} MB / {health['memory_limit_mb']:.0f} MB ({health['memory_usage_pct']:.1f}%)")
        print(f"    Cyclone Loaded: {health['cyclone_model_loaded']}")
        print(f"    Flood Loaded:   {health.get('flood_model_loaded', False)}")
        print(f"    Landslide:      {health.get('landslide_model_loaded', False)}")
        print(f"    Tsunami:        {health.get('tsunami_model_loaded', False)}")
    except Exception as e:
        print(f"    Health check failed (ensure server is running at {client.base_url}): {e}")
        return

    # 2. Multi-Hazard Registry Status
    print("\n[2] Checking Multi-Hazard Registry Status (GET /models)...")
    models_status = client.list_models()
    for hazard, info in models_status["models"].items():
        mode_tag = "PRODUCTION PROMOTED" if info["is_active"] else "DEMONSTRATION ONLY"
        print(f"    - {hazard.upper():<10}: {info['production_status']:<22} | {mode_tag}")

    # 3. Cyclone Forecast (Production Promoted)
    print("\n[3] Cyclone 24h Track & Intensity Forecast (Cyclone Remal - 2024-05-25 12:00 UTC)...")
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
        "storm_name": "REMAL"
    }
    cyc_res = client.predict_cyclone(remal_payload)
    p_cyc = cyc_res["predictions"]
    print(f"    Forecast (+24h): ({p_cyc['predicted_latitude']:.2f}°N, {p_cyc['predicted_longitude']:.2f}°E)")
    print(f"    Wind Speed:      {p_cyc['predicted_wind_speed_kts']:.1f} kts ({p_cyc['imd_intensity_category']})")
    print(f"    RI Alert:        {p_cyc['rapid_intensification_alert']} (prob: {p_cyc['rapid_intensification_probability']:.2f})")
    print(f"    Execution Mode:  {cyc_res.get('execution_mode', 'PRODUCTION')}")

    # 4. Flood Discharge Forecast (Demonstration Mode)
    print("\n[4] Flood 24h River Discharge Forecast (Baltara Gauging Station - Kosi Basin)...")
    flood_payload = {
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
    flood_res = client.predict_flood(flood_payload)
    p_fld = flood_res["predictions"]
    print(f"    Current Q (q_t):     {flood_payload['q_t']:.1f} m³/s")
    print(f"    Forecast Q (+24h):   {p_fld['predicted_discharge_m3_s']:.1f} m³/s (Delta: {p_fld['discharge_delta_pct']:+.1f}%)")
    print(f"    Severity Stage:     {p_fld['estimated_flood_severity']}")
    print(f"    Execution Mode:     {flood_res['execution_mode']} ({flood_res['production_status']})")

    # 5. Landslide Susceptibility Screening (Demonstration Mode)
    print("\n[5] Landslide Susceptibility Screening (Wayanad Hills Analogue)...")
    landslide_payload = {
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
    ls_res = client.predict_landslide(landslide_payload)
    p_ls = ls_res["predictions"]
    print(f"    Susceptibility Prob: {p_ls['susceptibility_probability']:.4f} ({p_ls['susceptibility_level']})")
    print(f"    High Risk Alert:     {p_ls['high_risk_alert']}")
    print(f"    Trigger Factors:     {', '.join(p_ls['critical_trigger_factors'][:2])}")
    print(f"    Execution Mode:      {ls_res['execution_mode']} ({ls_res['production_status']})")

    # 6. Tsunami Potential Early Screening (Demonstration Mode)
    print("\n[6] Tsunami Potential Early Screening (Indian Ocean Megathrust Analogue)...")
    tsunami_payload = {
        "event_id": "indian_ocean_analogue",
        "magnitude": 8.2,
        "focal_depth_km": 15.0,
        "epicenter_latitude": 3.3,
        "epicenter_longitude": 95.8,
        "gebco_bathymetry_m": -3500.0,
        "distance_to_nearest_coast_km": 45.0
    }
    ts_res = client.predict_tsunami(tsunami_payload)
    p_ts = ts_res["predictions"]
    print(f"    Tsunamigenic Prob:   {p_ts['tsunamigenic_probability']:.4f}")
    print(f"    Screening Alert:     {p_ts['tsunami_screening_alert']} (Threshold: {p_ts['decision_threshold']})")
    print(f"    Advisory Category:   {p_ts['screening_category']}")
    print(f"    Empirical Rule:      {p_ts['empirical_rule_triggered']}")
    print(f"    Execution Mode:      {ts_res['execution_mode']} ({ts_res['production_status']})")

    print("\n" + "=" * 75)
    print("Multi-hazard programmatic demonstration complete.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

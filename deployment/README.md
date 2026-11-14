# AAPADAMITRA – Zero-Cost Remote Multi-Hazard API Deployment & Demonstration Guide

This document provides complete instructions for deploying and querying the **AAPADAMITRA Multi-Hazard Disaster Monitoring & Early Warning Platform API** under a **strictly ₹0 budget**, using:
1. **GitHub**: Deployment branch hosting application code and verified model artifacts.
2. **Render Free Web Service**: Free FastAPI inference server with automated cold-start handling and 512 MB memory boundary.
3. **Multi-Hazard Demonstration Mode**: Enables all four existing hazard modules (Cyclone, Flood, Landslide, Tsunami) for academic/project demonstration with rigorous scientific audit caveats and statutory authority demarcation.
4. **Python Client SDK**: Client library for teammates to run predictions from local scripts or terminal CLI.

---

## 1. Multi-Hazard Architecture & Production vs Demonstration Demarcation

Following the Phase 3 standardized retraining and independent scientific audit:
- **Cyclone v3.0**: **PROMOTED TO PRODUCTION** (Outperforms operational persistence benchmark by 15.7%).
- **Flood v3.0**: **NOT PROMOTED** (Candidate NSE 0.9189 vs 1-day persistence benchmark 0.9229; enabled under **Demonstration Mode**).
- **Landslide v3.0**: **NOT PROMOTED** (Artifact of synthetic non-landslide background sampling separation; enabled under **Demonstration Mode**).
- **Tsunami v3.0**: **NOT PROMOTED** (Candidate PR-AUC 0.6500 lower than pre-existing baseline 0.6845; enabled under **Demonstration Mode**).

```
                                  STRICTLY ₹0 MULTI-HAZARD INFRASTRUCTURE
 +---------------------------------------------------------------------------------------------------------+
 |                                                                                                         |
 |  [GitHub Deployment Branch]                                [Render Web Service (Free Tier)]             |
 |  +-------------------------------------+                   +---------------------------------------+    |
 |  | Local Repository Model Artifacts    |                   | Python 3.11 + FastAPI Server          |    |
 |  |  • Cyclone v3.0 (4 boosters, 945KB) |== (Direct Git) ==>|  • Memory RSS: ~202 MB / 512 MB       |    |
 |  |  • Flood v3.0 (RF, 9.3 MB)          |    Checkout       |  • 4 Hazards In-Memory                |    |
 |  |  • Landslide v3.0 (Cal RF, 588 KB)  |  SHA256 Verified  |  • Cold-start wake: 30-50s            |    |
 |  |  • Tsunami v3.0 (RF+Scaler, 217 KB) |                   |  • Auth: X-API-Key Header             |    |
 |  |  • Total Model Disk: ~10.8 MB       |                   +-------------------+-------------------+    |
 |  +-------------------------------------+                                       ^                        |
 |                                                                                 | HTTPS (TLS)            |
 |                                                                        X-API-Key Authenticated           |
 |                                                                                 |                        |
 |                                                                +----------------+------------------+     |
 |                                                                | Remote Python Client SDK & CLI    |     |
 |                                                                | (Teammates, Streamlit, Postman)   |     |
 |                                                                +-----------------------------------+     |
 +---------------------------------------------------------------------------------------------------------+
```

### Strict Financial Boundaries
- **₹0 Spend**: No payment methods or credit cards required.
- **Direct Local Model Packaging**: All model artifacts (~10.8 MB total) are stored in the repository. No external storage (Hugging Face / S3 / GCS) is required.
- **No Paid Cloud Infrastructure**: Strictly no AWS, GCP, Azure, or paid Render tiers.
- **No Managed Databases or Caches**: Models run in-memory within Render's free 512 MB RAM container. No Redis or PostgreSQL required.

---

## 2. Multi-Hazard Model Technical Profiles & Verification

| Hazard Module | Model Identifier | Architecture | Validation Status | Demonstration Status | Key Metrics | Statutory Authority |
|:---|:---|:---|:---:|:---:|:---|:---|
| **Cyclone** | `cyclone_track_and_intensity_24h` | Multi-Task LightGBM (`v3.0`) | **PROMOTED** | `PRODUCTION_APPROVED` | Track error: **136.50 km** (Persistence: 161.84 km), RI PR-AUC: 0.2379 | India Meteorological Department (IMD) |
| **Flood** | `flood_discharge_rf` | RandomForestRegressor | **NOT PROMOTED** | `DEMONSTRATION_ONLY` | NSE: 0.9189 (Persistence: **0.9229**), KGE: 0.9394, MAE: 53.19 m³/s | Central Water Commission (CWC) / NWIC |
| **Landslide** | `landslide_susceptibility` | CalibratedClassifierCV | **NOT PROMOTED** | `DEMONSTRATION_ONLY` | PR-AUC: 1.0 (Synthetic background separation artifact) | Geological Survey of India (GSI) |
| **Tsunami** | `tsunami_classifier_rf` | RandomForestClassifier | **NOT PROMOTED** | `DEMONSTRATION_ONLY` | PR-AUC: 0.6500 (Pre-existing baseline: **0.6845**), F2 threshold: 0.44 | Indian National Centre for Ocean Information Services (INCOIS) |

---

## 3. Cryptographic SHA256 Checksums

The server validates every SHA256 checksum at startup. Any missing or modified artifact causes a fatal failure:

| Hazard | Filename | Disk Size | SHA256 Cryptographic Checksum |
|:---|:---|:---:|:---|
| **Cyclone** | `cyclone_track_lat_24h_v3.joblib` | 268.0 KB | `a06f8037a356f0ba8172b1d90c0fe069b7901d6bd1dfde79059e4974d4643fbd` |
| **Cyclone** | `cyclone_track_lon_24h_v3.joblib` | 257.5 KB | `2db98fee0fc19743aa35ef56cbd91786c908cd2d8415d31bf1167c5a41ac16dc` |
| **Cyclone** | `cyclone_intensity_24h_v3.joblib` | 249.6 KB | `0cc1e42e5480fd28745c0a9a9efdfc8f0b6d143f62d5f208011e016ead9b657e` |
| **Cyclone** | `cyclone_ri_classifier_24h_v3.joblib` | 169.8 KB | `e5b959b08bc15eb252f80d06d669d51842d692f2a5ed7601beae4a29ffe0b9c7` |
| **Flood** | `flood_discharge_rf_v3.joblib` | 9,313.4 KB | `4c251981abb76c7ed4cadf577e965ca8871149cf7b4f818416f1509c368a8ec5` |
| **Landslide**| `landslide_calibrated_rf_v3.joblib` | 587.7 KB | `ced3917ad499846633578d47bdec283df07a4b1fdafdd891318c050b093d303d` |
| **Tsunami** | `tsunami_classifier_rf_v3.joblib` | 215.6 KB | `739d9a7f52c3af067fff1ea31d6278713b0944678f2524ab821f5642eb5e501c` |
| **Tsunami** | `tsunami_scaler_v3.joblib` | 1.3 KB | `148d4a866860e95e8ebda65b63ceffc18325379d5a9af98532ef8d3728a8aa5f` |

---

## 4. Free-Tier Operational Characteristics

### 4.1 RAM Consumption & Safety Margins (Render Free Tier: 512 MB Limit)
- **FastAPI + Uvicorn Runtime**: ~107 MB RSS.
- **Cyclone Model Bundle (4 boosters)**: ~69 MB.
- **Flood Model Bundle (RF regressor)**: ~18 MB.
- **Landslide Model Bundle (Calibrated RF)**: ~4 MB.
- **Tsunami Model Bundle (RF + Scaler)**: ~4 MB.
- **Total Operational Memory with All 4 Models Loaded**: **~202 MB RSS** (39.5% of free quota).
- **Headroom Margin**: >310 MB buffer remaining, preventing Out-Of-Memory (OOM) eviction.

### 4.2 Cold-Start Behavior (Spin-Down)
- On Render Free Tier, services spin down after **15 minutes** of inactivity.
- Cold start wake time: ~30 to 50 seconds.
- The included Python client (`AapadamitraClient`) automatically retries with exponential backoff.

---

## 5. Endpoints Reference & Examples

### 5.1 Public Metadata & Health Endpoints
- `GET /health`: System health, RAM usage in MB, loaded status for all 4 models.
- `GET /models`: Multi-hazard registry status and promotion rationale.
- `GET /api/v1/hazards`: Complete catalog with documentation, endpoints, and statutory authorities.
- `GET /api/v1/cyclone/model-info`: Technical model card for Cyclone v3.0.
- `GET /api/v1/flood/model-info`: Technical model card for Flood v3.0.
- `GET /api/v1/landslide/model-info`: Technical model card for Landslide v3.0.
- `GET /api/v1/tsunami/model-info`: Technical model card for Tsunami v3.0.

### 5.2 Protected Prediction Endpoints (Require `X-API-Key`)

#### 1. Cyclone Forecast (Production)
```bash
curl -X POST "https://aapadamitra-model-api.onrender.com/api/v1/cyclone/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{
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
     }'
```

#### 2. Flood River Discharge Forecast (Demonstration)
```bash
curl -X POST "https://aapadamitra-model-api.onrender.com/api/v1/flood/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{
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
     }'
```

#### 3. Landslide Susceptibility Screening (Demonstration)
```bash
curl -X POST "https://aapadamitra-model-api.onrender.com/api/v1/landslide/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{
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
     }'
```

#### 4. Tsunami Potential Early Screening (Demonstration)
```bash
curl -X POST "https://aapadamitra-model-api.onrender.com/api/v1/tsunami/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{
       "event_id": "indian_ocean_analogue",
       "magnitude": 8.2,
       "focal_depth_km": 15.0,
       "epicenter_latitude": 3.3,
       "epicenter_longitude": 95.8,
       "gebco_bathymetry_m": -3500.0,
       "distance_to_nearest_coast_km": 45.0
     }'
```

---

## 6. Python Client SDK & Programmatic Usage

```python
from deployment.client.remote_client import AapadamitraClient

client = AapadamitraClient(
    base_url="https://aapadamitra-model-api.onrender.com",
    api_key="your-api-key"
)

# 1. Health check
health = client.health_check()
print(f"Status: {health['status']} | RAM: {health['memory_rss_mb']} MB")

# 2. Multi-hazard catalog
hazards = client.list_hazards()
print(f"Total hazards: {hazards['total_hazards']}")

# 3. Cyclone Forecast (Production)
cyc = client.predict_cyclone(lat=18.8, lon=89.4, wind=32.4, pressure=993.0, dist=314.0, speed=10.0, heading=315.0, sst=29.4, shear=16.2, rh=81.9)
print(f"Cyclone (+24h): {cyc['predictions']['predicted_latitude']}°N, {cyc['predictions']['predicted_longitude']}°E")

# 4. Flood Forecast (Demo)
flood = client.predict_flood(q=1250.0, q_lag1=1180.0, q_lag2=1100.0, q_lag3=1050.0, station="Baltara")
print(f"Forecast Q: {flood['predictions']['predicted_discharge_m3_s']} m³/s ({flood['predictions']['estimated_flood_severity']})")

# 5. Landslide Screening (Demo)
ls = client.predict_landslide(elevation=1240.0, slope=34.5, rain3d=135.0, rain7d=240.0, location="Wayanad")
print(f"Landslide Prob: {ls['predictions']['susceptibility_probability']:.4f} ({ls['predictions']['susceptibility_level']})")

# 6. Tsunami Screening (Demo)
ts = client.predict_tsunami(mag=8.2, depth=15.0, lat=3.3, lon=95.8, bathy=-3500.0, dist=45.0)
print(f"Tsunami Alert: {ts['predictions']['tsunami_screening_alert']} (prob: {ts['predictions']['tsunamigenic_probability']:.4f})")
```

---

## 7. Statutory Early Warning Authority Demarcation

AAPADAMITRA models are computational scientific decision-support tools. Official statutory early warnings in India are governed exclusively by national nodal agencies:
- **Tropical Cyclone**: India Meteorological Department (IMD) - RSMC New Delhi.
- **River Flood & Stage Hydrographs**: Central Water Commission (CWC) / NWIC.
- **Landslides & Slope Stability**: Geological Survey of India (GSI).
- **Tsunami Early Warning**: Indian National Centre for Ocean Information Services (INCOIS) - ITEWC.

Do **NOT** use experimental model predictions for operational life-safety evacuation decisions without consulting official bulletins from the respective statutory authorities.

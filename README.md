# AAPADAMITRA – Production Cyclone Inference API

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Hosting: Render Free Tier](https://img.shields.io/badge/Render-Free%20Tier%20(₹0)-purple.svg)](https://render.com)
[![Status: Production Approved](https://img.shields.io/badge/Status-Production%20Approved-success.svg)](#)

Production inference deployment branch for **AAPADAMITRA**, an AI-powered Multi-Hazard Early Warning Decision Support Platform.

> **Operational Notice**: In strict accordance with the Phase 3 standardized retraining and scientific validation audit, **only the Tropical Cyclone Model (`v3.0`) is approved and promoted to production**. Unpromoted hazard modules (Flood, Landslide, Tsunami) are blocked with HTTP 403 Forbidden.

---

## 1. Deployed Model Specifications

- **Hazard Domain**: Tropical Cyclone Track, Intensity & Rapid Intensification
- **Model Version**: `v3.0` (Production Promoted)
- **Forecast Horizon**: +24 Hours
- **Architecture**: Multi-Task LightGBM Regressors (`dlat`, `dlon`, `dwind`) + LightGBM Binary Classifier (Rapid Intensification)
- **Performance Benchmark (2021–2024 Holdout)**:
  - Mean Track Geodesic Error: **136.50 km** (15.7% error reduction over dead-reckoning persistence: 161.84 km)
  - Intensity MAE: **11.30 knots** (Beats persistence: 14.50 knots)
  - Rapid Intensification PR-AUC: **0.2379** (~3x climatological base rate)

### Included Verified Model Artifacts
All model artifacts are checked in directly under `models/training/cyclone/artifacts/` and cryptographically verified on startup:

| Artifact File | Size | SHA256 Cryptographic Digest |
|:---|:---:|:---|
| `cyclone_track_lat_24h_v3.joblib` | ~268 KB | `a06f8037a356f0ba8172b1d90c0fe069b7901d6bd1dfde79059e4974d4643fbd` |
| `cyclone_track_lon_24h_v3.joblib` | ~258 KB | `2db98fee0fc19743aa35ef56cbd91786c908cd2d8415d31bf1167c5a41ac16dc` |
| `cyclone_intensity_24h_v3.joblib` | ~250 KB | `0cc1e42e5480fd28745c0a9a9efdfc8f0b6d143f62d5f208011e016ead9b657e` |
| `cyclone_ri_classifier_24h_v3.joblib` | ~170 KB | `e5b959b08bc15eb252f80d06d669d51842d692f2a5ed7601beae4a29ffe0b9c7` |

Total artifact bundle size is **~945 KB** (<1 MB). Memory consumption with all 4 models loaded is **175 MB RSS** (well below Render's 512 MB free tier ceiling).

---

## 2. Render Free Web Service Deployment (₹0)

1. Connect this repository to [Render](https://render.com).
2. Create a new **Web Service**:
   - **Branch**: `deploy/render-cyclone-api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn deployment.api.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Plan**: `Free` (512 MB RAM, ₹0)
3. Set Environment Variables:
   - `PYTHON_VERSION`: `3.11.9`
   - `ENVIRONMENT`: `production`
   - `MODEL_DIR`: `models/training/cyclone/artifacts`
   - `API_KEY`: *(Generate secure random key in Render)*
4. Deploy!

---

## 3. Endpoints

- `GET /health`: Server health, uptime, and real-time process memory consumption.
- `GET /models`: Multi-hazard promotion registry (Cyclone: ACTIVE; Flood, Landslide, Tsunami: NOT PROMOTED).
- `GET /api/v1/cyclone/model-info`: Detailed model card, feature requirements, and validation benchmarks.
- `POST /api/v1/cyclone/predict`: Authenticated 24-hour prediction (`X-API-Key` required).

---

## 4. Teammate Python Client

```python
from deployment.client.remote_client import AapadamitraClient

client = AapadamitraClient(
    base_url="https://aapadamitra-model-api.onrender.com",
    api_key="your-api-key"
)

# Predict 24h cyclone track and intensity
forecast = client.predict_cyclone(
    lat=18.8,
    lon=89.4,
    wind=32.4,
    pressure=993.0,
    dist=314.0,
    speed=10.0,
    heading=315.0,
    sst=29.4,
    shear=16.2,
    rh=81.9,
    name="REMAL"
)
print(forecast["predictions"])
```

---

## 5. Statutory Early Warning Notice

**STATUTORY DEMARCATION**: AAPADAMITRA tropical cyclone models are experimental computational aids. Official cyclone warnings, tracks, landfall bulletins, and evacuation orders for the North Indian Ocean are issued exclusively by the **India Meteorological Department (IMD) - Regional Specialized Meteorological Centre (RSMC) New Delhi**.

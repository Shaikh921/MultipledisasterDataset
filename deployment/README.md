# AAPADAMITRA – Zero-Cost Remote Model API Deployment Guide

This document provides complete instructions for deploying and querying the **AAPADAMITRA Tropical Cyclone 24-Hour Prediction API** under a **strictly ₹0 budget**, using:
1. **GitHub**: Dedicated, lightweight deployment branch hosting application code and the 4 verified LightGBM model boosters (<1 MB total).
2. **Render Free Web Service**: Free FastAPI inference server with automated cold-start handling and 512 MB memory boundary.
3. **Python Client SDK**: Client library for teammates to run predictions from local scripts or terminal CLI.

---

## 1. Architecture & Strict ₹0 Budget Demarcation

```
                                  STRICTLY ₹0 INFRASTRUCTURE
 +-----------------------------------------------------------------------------------------+
 |                                                                                         |
 |  [GitHub Deployment Branch]                         [Render Web Service (Free Tier)]     |
 |  +-----------------------------+                    +-------------------------------+   |
 |  | Lightweight Deploy Branch   |                    | Python 3.11 + FastAPI Server  |   |
 |  |  • cyclone_track_lat_24h    |=== (Direct Git) ==>|  • Memory RSS: ~175 MB / 512MB|   |
 |  |  • cyclone_track_lon_24h    |     Checkout       |  • Cold start wake: 30-50s    |   |
 |  |  • cyclone_intensity_24h    |   SHA256 Verified  |  • Port: $PORT (0.0.0.0)      |   |
 |  |  • cyclone_ri_classifier    |                    |  • Auth: X-API-Key Header     |   |
 |  +-----------------------------+                    +---------------+---------------+   |
 |                                                                     ^                   |
 |                                                                     | HTTPS (TLS)       |
 |                                                            X-API-Key Authenticated      |
 |                                                                     |                   |
 |                                                     +---------------+---------------+   |
 |                                                     | Remote Python Client / CLI    |   |
 |                                                     | (Teammates & Downstream Apps) |   |
 |                                                     +-------------------------------+   |
 +-----------------------------------------------------------------------------------------+
```

### Strict Financial Boundaries
- **₹0 Spend**: No payment methods or credit cards required.
- **Direct Model Packaging**: The 4 approved LightGBM model boosters (~945 KB total) are committed directly to the dedicated deployment branch. No external storage (Hugging Face / S3 / GCS) is required.
- **No Paid Cloud Infrastructure**: Strictly no AWS, GCP, Azure, or paid Render tiers.
- **No Managed Databases or Caches**: Models run in-memory within Render's free 512 MB RAM container. No Redis or PostgreSQL required.

---

## 2. Production Promotion Audit Status

Following the Phase 3 standardized retraining and scientific validation audit, only the **Tropical Cyclone Module (`v3.0`)** is approved for production deployment.

| Hazard Module | Candidate Architecture | Phase 3 Audit Decision | API Route Status | Promotion Rationale |
|:---|:---|:---:|:---:|:---|
| **Cyclone** | Multi-Task LightGBM (`v3.0`) | **PROMOTED** | `ACTIVE` (`POST /api/v1/cyclone/predict`) | Mean 24h track error (**136.50 km**) beats persistence benchmark (**161.84 km**) by 15.7%. |
| **Flood** | Random Forest Regressor | **REJECTED** | `BLOCKED` (HTTP 403) | Candidate NSE (0.9189) does not beat persistence benchmark (0.9229). Retained in research catalog. |
| **Landslide** | Calibrated Random Forest | **REJECTED** | `BLOCKED` (HTTP 403) | Candidate PR-AUC (1.0) does not outperform baseline (1.0). Retained in research catalog. |
| **Tsunami** | Random Forest Classifier | **REJECTED** | `BLOCKED` (HTTP 403) | Candidate PR-AUC (0.6500) lower than pre-existing baseline (0.6845). Retained in research catalog. |

---

## 3. Approved Model Artifacts & Cryptographic Verification

All 4 approved LightGBM models reside at `models/training/cyclone/artifacts/`:

| Filename | Disk Size | SHA256 Cryptographic Checksum | Purpose |
|:---|:---:|:---|:---|
| `cyclone_track_lat_24h_v3.joblib` | ~268 KB | `a06f8037a356f0ba8172b1d90c0fe069b7901d6bd1dfde79059e4974d4643fbd` | Latitude displacement (+24h) |
| `cyclone_track_lon_24h_v3.joblib` | ~258 KB | `2db98fee0fc19743aa35ef56cbd91786c908cd2d8415d31bf1167c5a41ac16dc` | Longitude displacement (+24h) |
| `cyclone_intensity_24h_v3.joblib` | ~250 KB | `0cc1e42e5480fd28745c0a9a9efdfc8f0b6d143f62d5f208011e016ead9b657e` | Intensity displacement (+24h) |
| `cyclone_ri_classifier_24h_v3.joblib` | ~170 KB | `e5b959b08bc15eb252f80d06d669d51842d692f2a5ed7601beae4a29ffe0b9c7` | Rapid Intensification probability |

The model loader strictly validates every SHA256 hash upon startup; any missing file or corrupted hash causes an immediate fatal startup failure (`ValueError`).

---

## 4. Render Free Web Service Deployment (₹0)

### 4.1 Deploying the Web Service on Render
1. Sign in to [https://render.com](https://render.com) (100% Free Plan).
2. Connect your GitHub repository: `https://github.com/Shaikh921/MultipledisasterDataset`.
3. Click **New +** -> **Web Service**.
4. Configure the service:
   - **Name**: `aapadamitra-model-api`
   - **Region**: Singapore (lowest latency to India)
   - **Branch**: Dedicated deployment branch (e.g. `deploy/render-cyclone-api`)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r deployment/requirements.txt`
   - **Start Command**: `uvicorn deployment.api.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Plan**: `Free` (512 MB RAM, 0.1 CPU, ₹0)

### 4.2 Configure Environment Variables
In the Render dashboard under **Environment**:

| Key | Value | Description |
|:---|:---|:---|
| `PYTHON_VERSION` | `3.11.9` | Pinned Python runtime |
| `ENVIRONMENT` | `production` | Production mode |
| `MODEL_DIR` | `models/training/cyclone/artifacts` | Path to local model artifacts |
| `API_KEY` | *(Click "Generate" or set secure random hex)* | Key required in `X-API-Key` header |

### 4.3 Click "Deploy Web Service"
Render builds the container, initializes Uvicorn, verifies SHA256 checksums, and exposes your HTTPS endpoint (e.g. `https://aapadamitra-model-api.onrender.com`).

---

## 5. Free-Tier Operational Characteristics

### 5.1 RAM Consumption & Safety Margins
- **Render Free Ceiling**: 512.0 MB Resident Set Size (RSS).
- **FastAPI + Uvicorn Runtime**: ~105 MB RSS.
- **Model Bundle Footprint (4 LightGBM Boosters)**: ~69 MB RSS.
- **Total Operational Memory**: **~175 MB RSS** (34.2% of free quota).
- **Headroom Margin**: >330 MB safety margin, completely preventing Out-Of-Memory (OOM) termination.

### 5.2 Cold-Start Behavior (Spin-Down)
- On Render's Free tier, services with no incoming traffic spin down after **15 minutes**.
- Cold start wake time: ~30 to 50 seconds.
- The included Python client (`AapadamitraClient`) features **automatic retry with exponential backoff** to handle cold starts transparently.

---

## 6. API Reference & cURL Examples

### 6.1 `GET /health` (Public)
Checks server health, uptime, and real-time process memory consumption.

```bash
curl -X GET "https://aapadamitra-model-api.onrender.com/health"
```

### 6.2 `GET /models` (Public)
Lists all four hazard modules and their production audit promotion decisions.

```bash
curl -X GET "https://aapadamitra-model-api.onrender.com/models"
```

### 6.3 `GET /api/v1/cyclone/model-info` (Public)
Returns full scientific metadata, feature requirements, holdout validation protocol, and benchmark scores.

```bash
curl -X GET "https://aapadamitra-model-api.onrender.com/api/v1/cyclone/model-info"
```

### 6.4 `POST /api/v1/cyclone/predict` (Protected by `X-API-Key`)
Executes 24-hour track displacement, intensity forecasting, and rapid intensification probability calculation.

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
       "storm_name": "REMAL",
       "timestamp_utc": "2024-05-25T12:00:00Z"
     }'
```

---

## 7. Python Client SDK for Teammates

### 7.1 Programmatic Python Usage
```python
from deployment.client.remote_client import AapadamitraClient

# Initialize client
client = AapadamitraClient(
    base_url="https://aapadamitra-model-api.onrender.com",
    api_key="your-api-key"
)

# 1. Health check
health = client.health_check()
print(f"Server Status: {health['status']} | RAM: {health['memory_rss_mb']} MB")

# 2. Predict 24h Cyclone Forecast
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

preds = forecast["predictions"]
print(f"Predicted Position (+24h): ({preds['predicted_latitude']}°N, {preds['predicted_longitude']}°E)")
print(f"Predicted Intensity:      {preds['predicted_wind_speed_kts']} kts ({preds['imd_intensity_category']})")
print(f"RI Warning Alert:         {preds['rapid_intensification_alert']}")
```

### 7.2 Terminal CLI Usage
```bash
python deployment/client/remote_client.py \
  --url https://aapadamitra-model-api.onrender.com \
  --api-key your-api-key \
  predict \
  --lat 18.8 --lon 89.4 --wind 32.4 --pressure 993.0 --dist 314.0 \
  --speed 10.0 --heading 315.0 --sst 29.4 --shear 16.2 --rh 81.9 --storm-name REMAL
```

---

## 8. Statutory Disclaimer

**STATUTORY EARLY WARNING AUTHORITY DEMARCATION**:
AAPADAMITRA tropical cyclone models are experimental scientific decision-support computational aids. All official cyclone tracks, landfall predictions, intensity advisories, and public early warnings for the North Indian Ocean basin (Bay of Bengal and Arabian Sea) are issued exclusively by the **India Meteorological Department (IMD) - Regional Specialized Meteorological Centre (RSMC) New Delhi**. This model must **never** be used to supersede official IMD bulletins or direct operational life-safety evacuation procedures.

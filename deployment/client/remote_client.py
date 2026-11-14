"""
AAPADAMITRA Remote API Python Client SDK
Lightweight, resilient client for remote inference on Render Free Web Service.
Handles cold starts with exponential backoff, API key authentication,
and structured prediction formatting across all four hazard modules:
1. Tropical Cyclone (Production)
2. River Flood (Demonstration)
3. Landslide Susceptibility (Demonstration)
4. Tsunami Potential (Demonstration)
"""

from typing import Dict, Any, Optional
import os
import sys
import time
import argparse
import requests
import json
import logging

logger = logging.getLogger("AapadamitraClient")


class AapadamitraClient:
    """Client for querying the deployed AAPADAMITRA FastAPI Multi-Hazard Inference API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "aapadamitra-dev-key",
        timeout: float = 45.0,
        max_retries: int = 3,
        retry_delay: float = 3.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Aapadamitra-PythonClient/3.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def _request_with_retry(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        auth_required: bool = False
    ) -> requests.Response:
        """
        Executes HTTP requests with retry logic to gracefully accommodate
        Render free tier cold starts (where instances sleep after 15 min inactivity).
        """
        url = f"{self.base_url}{path}"
        headers = {}
        if auth_required:
            headers["X-API-Key"] = self.api_key

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=headers,
                    timeout=self.timeout
                )
                if resp.status_code in [502, 503, 504] and attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Server returned HTTP {resp.status_code} (cold start wake-up). "
                        f"Retrying in {backoff:.1f}s (Attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(backoff)
                    continue

                return resp

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Connection attempt {attempt} failed ({e}). "
                        f"Render service may be waking up. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    raise ConnectionError(
                        f"Failed to connect to {self.base_url} after {self.max_retries} attempts. "
                        f"Verify server URL and that the service is running. Error: {e}"
                    ) from e

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected request termination.")

    # --------------------------------------------------------------------------
    # HEALTH & REGISTRY METHODS
    # --------------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Queries server health, memory consumption, and model loading status."""
        resp = self._request_with_retry("GET", "/health", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> Dict[str, Any]:
        """Queries deployment status and promotion decisions for all hazard modules."""
        resp = self._request_with_retry("GET", "/models", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def list_hazards(self) -> Dict[str, Any]:
        """Queries complete multi-hazard catalog including demonstration endpoints."""
        resp = self._request_with_retry("GET", "/api/v1/hazards", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # CYCLONE METHODS (PRODUCTION PROMOTED)
    # --------------------------------------------------------------------------
    def get_cyclone_model_info(self) -> Dict[str, Any]:
        """Fetches technical model card and evaluation metrics for Cyclone v3.0."""
        resp = self._request_with_retry("GET", "/api/v1/cyclone/model-info", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def predict_cyclone(
        self,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Executes a 24-hour Cyclone Forecast (Production)."""
        payload = dict(data) if data else {}
        payload.update(kwargs)

        aliases = {
            "lat": "current_latitude",
            "lon": "current_longitude",
            "wind": "current_wind_speed_kts_3min",
            "wind_speed": "current_wind_speed_kts_3min",
            "pressure": "current_central_pressure_hpa",
            "speed": "translation_speed_kts",
            "heading": "heading_degrees",
            "sst": "sea_surface_temperature_c",
            "shear": "vertical_wind_shear_kts",
            "rh": "relative_humidity_700hpa_pct",
            "dist": "dist2land_km",
            "name": "storm_name"
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)

        resp = self._request_with_retry(
            method="POST",
            path="/api/v1/cyclone/predict",
            json_data=payload,
            auth_required=True
        )

        if resp.status_code == 401:
            raise PermissionError(f"Authentication failed: {resp.text}")
        elif resp.status_code == 422:
            raise ValueError(f"Input validation error: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # FLOOD METHODS (DEMONSTRATION ONLY)
    # --------------------------------------------------------------------------
    def get_flood_model_info(self) -> Dict[str, Any]:
        """Fetches model card and benchmark metrics for candidate Flood discharge model."""
        resp = self._request_with_retry("GET", "/api/v1/flood/model-info", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def predict_flood(
        self,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Executes 24-hour River Discharge Forecast under Demonstration Mode."""
        payload = dict(data) if data else {}
        payload.update(kwargs)

        aliases = {
            "q": "q_t",
            "discharge": "q_t",
            "station": "station_name",
            "river": "river_name",
            "zero_gauge": "zero_gauge_m",
            "catchment": "catchment_sqkm",
            "doy": "day_of_year"
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)

        resp = self._request_with_retry(
            method="POST",
            path="/api/v1/flood/predict",
            json_data=payload,
            auth_required=True
        )

        if resp.status_code == 401:
            raise PermissionError(f"Authentication failed: {resp.text}")
        elif resp.status_code == 422:
            raise ValueError(f"Input validation error: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # LANDSLIDE METHODS (DEMONSTRATION ONLY)
    # --------------------------------------------------------------------------
    def get_landslide_model_info(self) -> Dict[str, Any]:
        """Fetches model card and audit findings for candidate Landslide model."""
        resp = self._request_with_retry("GET", "/api/v1/landslide/model-info", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def predict_landslide(
        self,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Executes Landslide Susceptibility Screening under Demonstration Mode."""
        payload = dict(data) if data else {}
        payload.update(kwargs)

        aliases = {
            "elevation": "elevation_dem_m",
            "slope": "slope_degrees",
            "aspect": "aspect_degrees",
            "curvature": "profile_curvature",
            "twi": "topographic_wetness_index",
            "soil_depth": "soil_thickness_m",
            "rain3d": "antecedent_rainfall_3d_mm",
            "rain7d": "antecedent_rainfall_7d_mm",
            "moisture_surface": "soil_moisture_0_7cm_m3m3",
            "moisture_rootzone": "soil_moisture_rootzone_m3m3",
            "location": "location_name"
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)

        resp = self._request_with_retry(
            method="POST",
            path="/api/v1/landslide/predict",
            json_data=payload,
            auth_required=True
        )

        if resp.status_code == 401:
            raise PermissionError(f"Authentication failed: {resp.text}")
        elif resp.status_code == 422:
            raise ValueError(f"Input validation error: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # TSUNAMI METHODS (DEMONSTRATION ONLY)
    # --------------------------------------------------------------------------
    def get_tsunami_model_info(self) -> Dict[str, Any]:
        """Fetches model card and benchmark metrics for candidate Tsunami model."""
        resp = self._request_with_retry("GET", "/api/v1/tsunami/model-info", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def predict_tsunami(
        self,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Executes Tsunami Potential Early Screening under Demonstration Mode."""
        payload = dict(data) if data else {}
        payload.update(kwargs)

        aliases = {
            "mag": "magnitude",
            "depth": "focal_depth_km",
            "lat": "epicenter_latitude",
            "lon": "epicenter_longitude",
            "bathy": "gebco_bathymetry_m",
            "dist": "distance_to_nearest_coast_km"
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)

        resp = self._request_with_retry(
            method="POST",
            path="/api/v1/tsunami/predict",
            json_data=payload,
            auth_required=True
        )

        if resp.status_code == 401:
            raise PermissionError(f"Authentication failed: {resp.text}")
        elif resp.status_code == 422:
            raise ValueError(f"Input validation error: {resp.text}")
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AAPADAMITRA Multi-Hazard Remote API CLI Client"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("AAPADAMITRA_API_URL", "http://localhost:8000"),
        help="Base URL of deployed API (e.g. https://aapadamitra-api.onrender.com)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("AAPADAMITRA_API_KEY", "aapadamitra-dev-key"),
        help="API authentication key (or set AAPADAMITRA_API_KEY env var)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: health
    subparsers.add_parser("health", help="Check server health and memory footprint")

    # Command: models
    subparsers.add_parser("models", help="List active vs unpromoted hazard models")

    # Command: hazards
    subparsers.add_parser("hazards", help="List multi-hazard catalog and endpoints")

    # Command: info
    p_info = subparsers.add_parser("info", help="Get technical model metadata and metrics")
    p_info.add_argument(
        "--hazard",
        choices=["cyclone", "flood", "landslide", "tsunami"],
        default="cyclone",
        help="Hazard module to inspect (default: cyclone)"
    )

    # Command: predict-cyclone
    p_cyc = subparsers.add_parser("predict-cyclone", help="Execute 24h cyclone prediction (Production)")
    p_cyc.add_argument("--lat", type=float, required=True, help="Current latitude (-90 to 90)")
    p_cyc.add_argument("--lon", type=float, required=True, help="Current longitude (-180 to 180)")
    p_cyc.add_argument("--wind", type=float, required=True, help="Wind speed (knots, 3-min)")
    p_cyc.add_argument("--pressure", type=float, required=True, help="Central pressure (hPa)")
    p_cyc.add_argument("--dist2land", type=float, required=True, help="Distance to land (km)")
    p_cyc.add_argument("--speed", type=float, required=True, help="Translation speed (kts)")
    p_cyc.add_argument("--heading", type=float, required=True, help="Heading degrees (0-360)")
    p_cyc.add_argument("--sst", type=float, required=True, help="Sea Surface Temp (C)")
    p_cyc.add_argument("--shear", type=float, required=True, help="Vertical wind shear (kts)")
    p_cyc.add_argument("--rh", type=float, required=True, help="Relative humidity 700hPa (%)")
    p_cyc.add_argument("--storm-name", type=str, default="UNKNOWN", help="Storm name")

    # Command: predict-flood
    p_fld = subparsers.add_parser("predict-flood", help="Execute 24h river discharge forecast (Demo)")
    p_fld.add_argument("--q", type=float, required=True, help="Current discharge q_t in m³/s")
    p_fld.add_argument("--q1", type=float, required=True, help="Lag-1 discharge (t-1) in m³/s")
    p_fld.add_argument("--q2", type=float, required=True, help="Lag-2 discharge (t-2) in m³/s")
    p_fld.add_argument("--q3", type=float, required=True, help="Lag-3 discharge (t-3) in m³/s")
    p_fld.add_argument("--zero-gauge", type=float, default=50.0, help="Station datum elevation (m)")
    p_fld.add_argument("--catchment", type=float, default=10000.0, help="Catchment area (km²)")
    p_fld.add_argument("--station", type=str, default="Baltara", help="Station name")

    # Command: predict-landslide
    p_ls = subparsers.add_parser("predict-landslide", help="Execute landslide screening (Demo)")
    p_ls.add_argument("--elevation", type=float, required=True, help="DEM elevation (m)")
    p_ls.add_argument("--slope", type=float, required=True, help="Slope in degrees (0-90)")
    p_ls.add_argument("--aspect", type=float, default=180.0, help="Aspect in degrees (0-360)")
    p_ls.add_argument("--curvature", type=float, default=0.0, help="Profile curvature")
    p_ls.add_argument("--twi", type=float, default=6.0, help="Topographic Wetness Index")
    p_ls.add_argument("--soil-depth", type=float, default=1.5, help="Soil depth (m)")
    p_ls.add_argument("--rain3d", type=float, required=True, help="3-day antecedent rainfall (mm)")
    p_ls.add_argument("--rain7d", type=float, required=True, help="7-day antecedent rainfall (mm)")
    p_ls.add_argument("--moisture-top", type=float, default=0.35, help="Topsoil moisture (0-1)")
    p_ls.add_argument("--moisture-root", type=float, default=0.35, help="Rootzone moisture (0-1)")
    p_ls.add_argument("--location", type=str, default="Western Ghats", help="Location name")

    # Command: predict-tsunami
    p_ts = subparsers.add_parser("predict-tsunami", help="Execute tsunami screening (Demo)")
    p_ts.add_argument("--mag", type=float, required=True, help="Moment magnitude Mw (4-10)")
    p_ts.add_argument("--depth", type=float, required=True, help="Focal depth in km (0-700)")
    p_ts.add_argument("--lat", type=float, required=True, help="Epicenter latitude")
    p_ts.add_argument("--lon", type=float, required=True, help="Epicenter longitude")
    p_ts.add_argument("--bathy", type=float, required=True, help="Bathymetry depth in m (negative)")
    p_ts.add_argument("--dist", type=float, required=True, help="Distance to coast in km")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    client = AapadamitraClient(base_url=args.url, api_key=args.api_key)

    try:
        if args.command == "health":
            res = client.health_check()
            print(json.dumps(res, indent=2))

        elif args.command == "models":
            res = client.list_models()
            print(json.dumps(res, indent=2))

        elif args.command == "hazards":
            res = client.list_hazards()
            print(json.dumps(res, indent=2))

        elif args.command == "info":
            if args.hazard == "cyclone":
                res = client.get_cyclone_model_info()
            elif args.hazard == "flood":
                res = client.get_flood_model_info()
            elif args.hazard == "landslide":
                res = client.get_landslide_model_info()
            elif args.hazard == "tsunami":
                res = client.get_tsunami_model_info()
            print(json.dumps(res, indent=2))

        elif args.command == "predict-cyclone":
            res = client.predict_cyclone(
                current_latitude=args.lat,
                current_longitude=args.lon,
                current_wind_speed_kts_3min=args.wind,
                current_central_pressure_hpa=args.pressure,
                dist2land_km=args.dist2land,
                translation_speed_kts=args.speed,
                heading_degrees=args.heading,
                sea_surface_temperature_c=args.sst,
                vertical_wind_shear_kts=args.shear,
                relative_humidity_700hpa_pct=args.rh,
                storm_name=args.storm_name
            )
            print(json.dumps(res, indent=2))

        elif args.command == "predict-flood":
            res = client.predict_flood(
                q_t=args.q,
                q_lag1=args.q1,
                q_lag2=args.q2,
                q_lag3=args.q3,
                zero_gauge_m=args.zero_gauge,
                catchment_sqkm=args.catchment,
                station_name=args.station
            )
            print(json.dumps(res, indent=2))

        elif args.command == "predict-landslide":
            res = client.predict_landslide(
                elevation_dem_m=args.elevation,
                slope_degrees=args.slope,
                aspect_degrees=args.aspect,
                profile_curvature=args.curvature,
                topographic_wetness_index=args.twi,
                soil_thickness_m=args.soil_depth,
                antecedent_rainfall_3d_mm=args.rain3d,
                antecedent_rainfall_7d_mm=args.rain7d,
                soil_moisture_0_7cm_m3m3=args.moisture_top,
                soil_moisture_rootzone_m3m3=args.moisture_root,
                location_name=args.location
            )
            print(json.dumps(res, indent=2))

        elif args.command == "predict-tsunami":
            res = client.predict_tsunami(
                magnitude=args.mag,
                focal_depth_km=args.depth,
                epicenter_latitude=args.lat,
                epicenter_longitude=args.lon,
                gebco_bathymetry_m=args.bathy,
                distance_to_nearest_coast_km=args.dist
            )
            print(json.dumps(res, indent=2))

    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

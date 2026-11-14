"""
AAPADAMITRA Remote API Python Client SDK
Lightweight, resilient client for remote inference on Render Free Web Service.
Handles cold starts with exponential backoff, API key authentication,
and structured prediction formatting.
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
    """Client for querying the deployed AAPADAMITRA FastAPI Inference API."""

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
                # If server is spinning up (502/503/504), wait and retry
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

    def health_check(self) -> Dict[str, Any]:
        """Queries server health, memory consumption, and uptime."""
        resp = self._request_with_retry("GET", "/health", auth_required=False)
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> Dict[str, Any]:
        """Queries the status of all four hazard models in the registry."""
        resp = self._request_with_retry("GET", "/models", auth_required=False)
        resp.raise_for_status()
        return resp.json()

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
        """
        Executes a 24-hour Cyclone Forecast.
        Accepts a dictionary of features or keyword arguments.
        """
        payload = dict(data) if data else {}
        payload.update(kwargs)

        # Field alias mapping for teammate convenience
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


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AAPADAMITRA Remote API CLI Client"
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

    # Command: info
    subparsers.add_parser("info", help="Get Cyclone v3.0 model metadata and metrics")

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Execute 24h cyclone prediction")
    p_pred.add_argument("--lat", type=float, required=True, help="Current latitude (-90 to 90)")
    p_pred.add_argument("--lon", type=float, required=True, help="Current longitude (-180 to 180)")
    p_pred.add_argument("--wind", type=float, required=True, help="Wind speed (knots, 3-min)")
    p_pred.add_argument("--pressure", type=float, required=True, help="Central pressure (hPa)")
    p_pred.add_argument("--dist2land", type=float, required=True, help="Distance to land (km)")
    p_pred.add_argument("--speed", type=float, required=True, help="Translation speed (kts)")
    p_pred.add_argument("--heading", type=float, required=True, help="Heading degrees (0-360)")
    p_pred.add_argument("--sst", type=float, required=True, help="Sea Surface Temp (C)")
    p_pred.add_argument("--shear", type=float, required=True, help="Vertical wind shear (kts)")
    p_pred.add_argument("--rh", type=float, required=True, help="Relative humidity 700hPa (%)")
    p_pred.add_argument("--storm-name", type=str, default="UNKNOWN", help="Storm name")

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

        elif args.command == "info":
            res = client.get_cyclone_model_info()
            print(json.dumps(res, indent=2))

        elif args.command == "predict":
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
            print("\n" + "=" * 60)
            print(f"AAPADAMITRA 24h Cyclone Forecast: {res.get('storm_name', 'Unnamed')}")
            print("=" * 60)
            preds = res["predictions"]
            print(f"  Forecast Position (+24h): ({preds['predicted_latitude']:.2f}°N, {preds['predicted_longitude']:.2f}°E)")
            print(f"  Position Displacement:    dlat = {preds['delta_latitude']:+.2f}°, dlon = {preds['delta_longitude']:+.2f}°")
            print(f"  Forecast Intensity:       {preds['predicted_wind_speed_kts']:.1f} kts ({preds['delta_wind_speed_kts']:+.1f} kts)")
            print(f"  IMD Intensity Stage:      {preds['imd_intensity_category']}")
            print(f"  Rapid Intensification:    {preds['rapid_intensification_probability'] * 100:.1f}% probability")
            print(f"  RI Alert Status:          {'[!] CRITICAL RI WARNING' if preds['rapid_intensification_alert'] else 'Normal Evolution'}")
            print("-" * 60)
            print(f"  Reference Mean Track Error: {res['reference_performance']['mean_track_error_km']:.1f} km")
            print(f"  Generated at:             {res['generated_at']}")
            print("=" * 60 + "\n")

    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

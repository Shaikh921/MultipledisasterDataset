"""
Unit tests for AapadamitraClient Python SDK across all four hazard modules.
"""

from unittest.mock import patch, MagicMock
import pytest
from deployment.client.remote_client import AapadamitraClient


def test_client_initialization():
    """Verifies that client initializes with correct parameters and headers."""
    client = AapadamitraClient(base_url="https://api.example.com", api_key="test-key-123")
    assert client.base_url == "https://api.example.com"
    assert client.api_key == "test-key-123"
    assert client.session.headers.get("X-API-Key") is None
    assert "User-Agent" in client.session.headers


@patch("requests.Session.request")
def test_client_health_check(mock_request):
    """Verifies health_check method delegates to GET /health."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "HEALTHY", "memory_rss_mb": 175.0}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000")
    res = client.health_check()
    assert res["status"] == "HEALTHY"
    mock_request.assert_called_once()
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "http://localhost:8000/health"


@patch("requests.Session.request")
def test_client_list_hazards(mock_request):
    """Verifies list_hazards method delegates to GET /api/v1/hazards."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"total_hazards": 4, "hazards": []}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000")
    res = client.list_hazards()
    assert res["total_hazards"] == 4
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "http://localhost:8000/api/v1/hazards"


@patch("requests.Session.request")
def test_client_predict_cyclone_with_aliases(mock_request):
    """Verifies predict_cyclone maps friendly teammate aliases to full feature names."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "SUCCESS", "predictions": {"predicted_latitude": 20.43}}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000", api_key="secret-key")
    res = client.predict_cyclone(
        lat=18.8,
        lon=89.4,
        wind=32.4,
        pressure=993.0,
        dist=314.0,
        speed=10.0,
        heading=315.0,
        sst=29.4,
        shear=16.2,
        rh=81.9
    )
    assert res["status"] == "SUCCESS"
    mock_request.assert_called_once()
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://localhost:8000/api/v1/cyclone/predict"
    assert kwargs["headers"]["X-API-Key"] == "secret-key"
    payload = kwargs["json"]
    assert payload["current_latitude"] == 18.8
    assert payload["current_longitude"] == 89.4
    assert payload["current_wind_speed_kts_3min"] == 32.4
    assert payload["current_central_pressure_hpa"] == 993.0


@patch("requests.Session.request")
def test_client_predict_flood_with_aliases(mock_request):
    """Verifies predict_flood maps friendly aliases to required flood features."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "SUCCESS", "execution_mode": "DEMONSTRATION"}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000", api_key="secret-key")
    res = client.predict_flood(
        q=1250.0,
        q_lag1=1180.0,
        q_lag2=1100.0,
        q_lag3=1050.0,
        station="Baltara"
    )
    assert res["status"] == "SUCCESS"
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://localhost:8000/api/v1/flood/predict"
    payload = kwargs["json"]
    assert payload["q_t"] == 1250.0
    assert payload["station_name"] == "Baltara"


@patch("requests.Session.request")
def test_client_predict_landslide_with_aliases(mock_request):
    """Verifies predict_landslide maps friendly aliases to terrain features."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "SUCCESS", "execution_mode": "DEMONSTRATION"}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000", api_key="secret-key")
    res = client.predict_landslide(
        elevation=1240.0,
        slope=34.5,
        aspect=180.0,
        curvature=-0.05,
        twi=6.8,
        rain3d=135.0,
        rain7d=240.0,
        moisture_surface=0.44,
        moisture_rootzone=0.40
    )
    assert res["status"] == "SUCCESS"
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://localhost:8000/api/v1/landslide/predict"
    payload = kwargs["json"]
    assert payload["elevation_dem_m"] == 1240.0
    assert payload["slope_degrees"] == 34.5
    assert payload["antecedent_rainfall_3d_mm"] == 135.0


@patch("requests.Session.request")
def test_client_predict_tsunami_with_aliases(mock_request):
    """Verifies predict_tsunami maps friendly aliases to hypocenter features."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "SUCCESS", "execution_mode": "DEMONSTRATION"}
    mock_request.return_value = mock_resp

    client = AapadamitraClient(base_url="http://localhost:8000", api_key="secret-key")
    res = client.predict_tsunami(
        mag=8.2,
        depth=15.0,
        lat=3.3,
        lon=95.8,
        bathy=-3500.0,
        dist=45.0
    )
    assert res["status"] == "SUCCESS"
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://localhost:8000/api/v1/tsunami/predict"
    payload = kwargs["json"]
    assert payload["magnitude"] == 8.2
    assert payload["focal_depth_km"] == 15.0
    assert payload["gebco_bathymetry_m"] == -3500.0

"""
Unit tests for AapadamitraClient Python SDK.
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

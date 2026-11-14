"""
Unit tests for GET /api/v1/hazards multi-hazard catalog endpoint.
"""


def test_hazards_list_endpoint(client):
    """Verifies that GET /api/v1/hazards provides comprehensive catalog of all 4 hazard modules."""
    response = client.get("/api/v1/hazards")
    assert response.status_code == 200
    data = response.json()

    assert data["total_hazards"] == 4
    assert data["promoted_hazards_count"] == 1
    assert data["demonstration_hazards_count"] == 4
    assert data["operational_mode"] == "MULTI_HAZARD_DEMONSTRATION"
    assert "system_note" in data

    hazards = {item["hazard"]: item for item in data["hazards"]}
    assert set(hazards.keys()) == {"cyclone", "flood", "landslide", "tsunami"}

    # Cyclone verification
    cyc = hazards["cyclone"]
    assert cyc["production_status"] == "PROMOTED TO PRODUCTION"
    assert cyc["demo_status"] == "PRODUCTION_APPROVED"
    assert cyc["endpoints"]["predict"] == "/api/v1/cyclone/predict"
    assert "IMD" in cyc["statutory_authority"]

    # Flood verification
    fld = hazards["flood"]
    assert fld["production_status"] == "NOT PROMOTED"
    assert fld["demo_status"] == "DEMONSTRATION_ONLY"
    assert fld["endpoints"]["predict"] == "/api/v1/flood/predict"
    assert "CWC" in fld["statutory_authority"]

    # Landslide verification
    ls = hazards["landslide"]
    assert ls["production_status"] == "NOT PROMOTED"
    assert ls["demo_status"] == "DEMONSTRATION_ONLY"
    assert ls["endpoints"]["predict"] == "/api/v1/landslide/predict"
    assert "GSI" in ls["statutory_authority"]

    # Tsunami verification
    ts = hazards["tsunami"]
    assert ts["production_status"] == "NOT PROMOTED"
    assert ts["demo_status"] == "DEMONSTRATION_ONLY"
    assert ts["endpoints"]["predict"] == "/api/v1/tsunami/predict"
    assert "INCOIS" in ts["statutory_authority"]


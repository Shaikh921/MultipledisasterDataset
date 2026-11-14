"""
Unit tests for Local Model Loader and Cryptographic SHA256 Verification.
Verifies that models load from local checkout, checksum mismatches are fatal,
and no external cloud/HF dependencies are required.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from deployment.api.model_loader import (
    ModelLoader,
    model_loader,
    compute_sha256,
    EXPECTED_CHECKSUMS,
    REQUIRED_ARTIFACTS
)
from deployment.api.config import settings


def test_model_bundle_loading():
    """Verifies that the model loader successfully initializes all 4 models from local checkout."""
    bundle = model_loader.get_bundle()
    assert bundle is not None
    assert bundle.is_ready is True
    assert bundle.model_lat is not None
    assert bundle.model_lon is not None
    assert bundle.model_wind is not None
    assert bundle.model_ri is not None
    assert bundle.load_time_seconds >= 0.0
    assert len(bundle.checksums) == 4
    assert "local_repository" in bundle.source


def test_artifact_sha256_checksums():
    """
    Verifies that loaded model artifacts strictly match the exact cryptographic
    SHA256 digests established during Phase 3 standardized training across all 4 hazards.
    """
    # 1. Cyclone artifacts
    cyclone_bundle = model_loader.get_bundle()
    for filename in REQUIRED_ARTIFACTS:
        actual_digest = cyclone_bundle.checksums.get(filename)
        expected_digest = EXPECTED_CHECKSUMS[filename]
        assert actual_digest == expected_digest, f"Checksum mismatch for {filename}"

    # 2. Flood artifact
    flood_bundle = model_loader.get_flood_bundle()
    assert flood_bundle.checksums["flood_discharge_rf_v3.joblib"] == EXPECTED_CHECKSUMS["flood_discharge_rf_v3.joblib"]

    # 3. Landslide artifact
    landslide_bundle = model_loader.get_landslide_bundle()
    assert landslide_bundle.checksums["landslide_calibrated_rf_v3.joblib"] == EXPECTED_CHECKSUMS["landslide_calibrated_rf_v3.joblib"]

    # 4. Tsunami artifacts
    tsunami_bundle = model_loader.get_tsunami_bundle()
    assert tsunami_bundle.checksums["tsunami_classifier_rf_v3.joblib"] == EXPECTED_CHECKSUMS["tsunami_classifier_rf_v3.joblib"]
    assert tsunami_bundle.checksums["tsunami_scaler_v3.joblib"] == EXPECTED_CHECKSUMS["tsunami_scaler_v3.joblib"]


def test_load_all_models():
    """Verifies that load_all_models initializes all 4 hazard model bundles."""
    bundles = model_loader.load_all_models()
    assert set(bundles.keys()) == {"cyclone", "flood", "landslide", "tsunami"}
    for hazard, b in bundles.items():
        assert b is not None, f"Bundle for {hazard} was not loaded"
        assert b.is_ready is True, f"Bundle for {hazard} is not ready"


def test_resolved_model_dir():
    """Verifies that resolved_model_dir finds existing local artifacts."""
    resolved = settings.resolved_model_dir
    assert resolved.exists()
    for req in REQUIRED_ARTIFACTS:
        assert (resolved / req).exists(), f"Missing artifact {req} in {resolved}"


def test_compute_sha256_utility(tmp_path):
    """Verifies SHA256 computation utility with known content."""
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"aapadamitra-test-content")
    digest = compute_sha256(test_file)
    import hashlib
    expected = hashlib.sha256(b"aapadamitra-test-content").hexdigest()
    assert digest == expected


def test_missing_model_file_causes_fatal_startup_error(tmp_path):
    """Verifies that a missing model artifact causes a fatal FileNotFoundError."""
    loader = ModelLoader()
    with patch.object(settings, "MODEL_DIR", str(tmp_path)):
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_cyclone_models()
        assert "not found" in str(exc_info.value).lower()


def test_checksum_mismatch_causes_fatal_startup_error(tmp_path):
    """Verifies that a corrupted model artifact causes a fatal ValueError, aborting startup."""
    # Create fake files with mismatched contents
    for req in REQUIRED_ARTIFACTS:
        (tmp_path / req).write_bytes(b"corrupted-file-content")

    loader = ModelLoader()
    with patch.object(settings, "MODEL_DIR", str(tmp_path)):
        with pytest.raises(ValueError) as exc_info:
            loader.load_cyclone_models()
        assert "FATAL: Cryptographic checksum mismatch" in str(exc_info.value)


def test_no_huggingface_dependencies():
    """Verifies that Hugging Face environment variables and tokens are NOT required."""
    # Ensure config has no HF settings
    assert not hasattr(settings, "HF_TOKEN") or getattr(settings, "HF_TOKEN", None) is None
    assert not hasattr(settings, "HF_REPO_ID") or getattr(settings, "HF_REPO_ID", None) is None

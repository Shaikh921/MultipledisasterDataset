"""
AAPADAMITRA Model Loader
Loads approved production model artifacts directly from the local Git repository
with strict cryptographic SHA256 checksum verification and fatal failure guards.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import hashlib
import logging
import joblib

from deployment.api.config import settings

logger = logging.getLogger("ModelLoader")

# Strict SHA256 checksums established during Phase 3 standardized retraining audit
EXPECTED_CHECKSUMS: Dict[str, str] = {
    "cyclone_track_lat_24h_v3.joblib": "a06f8037a356f0ba8172b1d90c0fe069b7901d6bd1dfde79059e4974d4643fbd",
    "cyclone_track_lon_24h_v3.joblib": "2db98fee0fc19743aa35ef56cbd91786c908cd2d8415d31bf1167c5a41ac16dc",
    "cyclone_intensity_24h_v3.joblib": "0cc1e42e5480fd28745c0a9a9efdfc8f0b6d143f62d5f208011e016ead9b657e",
    "cyclone_ri_classifier_24h_v3.joblib": "e5b959b08bc15eb252f80d06d669d51842d692f2a5ed7601beae4a29ffe0b9c7"
}

REQUIRED_ARTIFACTS = [
    "cyclone_track_lat_24h_v3.joblib",
    "cyclone_track_lon_24h_v3.joblib",
    "cyclone_intensity_24h_v3.joblib",
    "cyclone_ri_classifier_24h_v3.joblib"
]


def compute_sha256(file_path: Path) -> str:
    """Calculates SHA256 hex digest of a file in 64KB blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class CycloneModelBundle:
    """In-memory bundle holding verified LightGBM model boosters and verification metadata."""

    def __init__(
        self,
        model_lat: Any,
        model_lon: Any,
        model_wind: Any,
        model_ri: Any,
        source: str,
        load_time_seconds: float,
        checksums: Dict[str, str]
    ):
        self.model_lat = model_lat
        self.model_lon = model_lon
        self.model_wind = model_wind
        self.model_ri = model_ri
        self.source = source
        self.load_time_seconds = load_time_seconds
        self.checksums = checksums

    @property
    def is_ready(self) -> bool:
        return all([
            self.model_lat is not None,
            self.model_lon is not None,
            self.model_wind is not None,
            self.model_ri is not None
        ])


class ModelLoader:
    """Manages local verification and in-memory loading of approved Cyclone models."""

    def __init__(self):
        self._bundle: Optional[CycloneModelBundle] = None

    def load_cyclone_models(self) -> CycloneModelBundle:
        """
        Loads the 4 production cyclone models directly from the repository checkout.
        Enforces exact SHA256 match. Treats any mismatch or missing artifact as fatal.
        """
        if self._bundle is not None:
            return self._bundle

        start_time = time.time()
        model_dir = settings.resolved_model_dir

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory '{model_dir}' does not exist. "
                "Ensure cyclone model artifacts are committed and available at the configured MODEL_DIR."
            )

        logger.info(f"Loading approved Cyclone model artifacts from: {model_dir}")
        artifact_paths: Dict[str, Path] = {}

        # 1. Verify existence of all 4 required artifacts
        for filename in REQUIRED_ARTIFACTS:
            path = model_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Fatal: Required model artifact '{filename}' not found in '{model_dir}'. "
                    "Inference service cannot start without all four verified model boosters."
                )
            artifact_paths[filename] = path

        # 2. Strict SHA256 Checksum Verification
        verified_checksums: Dict[str, str] = {}
        for filename, path in artifact_paths.items():
            digest = compute_sha256(path)
            expected = EXPECTED_CHECKSUMS.get(filename)
            if expected and digest != expected:
                raise ValueError(
                    f"FATAL: Cryptographic checksum mismatch for artifact '{filename}'. "
                    f"Actual SHA256: {digest} | Expected: {expected}. "
                    "Aborting startup to prevent corrupted model inference."
                )
            verified_checksums[filename] = digest
            logger.info(f"Verified {filename}: SHA256={digest[:16]}... (MATCH)")

        # 3. Load Models into Memory
        model_lat = joblib.load(artifact_paths["cyclone_track_lat_24h_v3.joblib"])
        model_lon = joblib.load(artifact_paths["cyclone_track_lon_24h_v3.joblib"])
        model_wind = joblib.load(artifact_paths["cyclone_intensity_24h_v3.joblib"])
        model_ri = joblib.load(artifact_paths["cyclone_ri_classifier_24h_v3.joblib"])

        load_time = round(time.time() - start_time, 3)
        source = f"local_repository ({model_dir})"
        logger.info(f"Successfully loaded 4 Cyclone models in {load_time}s from {source}.")

        self._bundle = CycloneModelBundle(
            model_lat=model_lat,
            model_lon=model_lon,
            model_wind=model_wind,
            model_ri=model_ri,
            source=source,
            load_time_seconds=load_time,
            checksums=verified_checksums
        )
        return self._bundle

    def get_bundle(self) -> CycloneModelBundle:
        """Returns the loaded bundle, or loads if not yet initialized."""
        if self._bundle is None:
            return self.load_cyclone_models()
        return self._bundle


# Global singleton instance
model_loader = ModelLoader()

"""
AAPADAMITRA Model Loader
Loads approved model artifacts directly from the local Git repository
with strict cryptographic SHA256 checksum verification and fatal failure guards.
Supports:
1. Cyclone v3.0 (Production Promoted LightGBM multi-task boosters)
2. Flood v3.0 (Demonstration River Discharge Random Forest Regressor)
3. Landslide v3.0 (Demonstration Terrain Calibrated Random Forest Classifier)
4. Tsunami v3.0 (Demonstration Seismic Hypocenter Random Forest & Scaler)
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import hashlib
import logging
import joblib

from deployment.api.config import settings

logger = logging.getLogger("ModelLoader")

# Strict SHA256 checksums established during Phase 3 standardized retraining audit
EXPECTED_CHECKSUMS: Dict[str, str] = {
    # Cyclone v3.0 boosters (Production Promoted)
    "cyclone_track_lat_24h_v3.joblib": "a06f8037a356f0ba8172b1d90c0fe069b7901d6bd1dfde79059e4974d4643fbd",
    "cyclone_track_lon_24h_v3.joblib": "2db98fee0fc19743aa35ef56cbd91786c908cd2d8415d31bf1167c5a41ac16dc",
    "cyclone_intensity_24h_v3.joblib": "0cc1e42e5480fd28745c0a9a9efdfc8f0b6d143f62d5f208011e016ead9b657e",
    "cyclone_ri_classifier_24h_v3.joblib": "e5b959b08bc15eb252f80d06d669d51842d692f2a5ed7601beae4a29ffe0b9c7",
    # Flood v3.0 Random Forest (Demonstration)
    "flood_discharge_rf_v3.joblib": "4c251981abb76c7ed4cadf577e965ca8871149cf7b4f818416f1509c368a8ec5",
    # Landslide v3.0 Calibrated RF (Demonstration)
    "landslide_calibrated_rf_v3.joblib": "ced3917ad499846633578d47bdec283df07a4b1fdafdd891318c050b093d303d",
    # Tsunami v3.0 RF & Scaler (Demonstration)
    "tsunami_classifier_rf_v3.joblib": "739d9a7f52c3af067fff1ea31d6278713b0944678f2524ab821f5642eb5e501c",
    "tsunami_scaler_v3.joblib": "148d4a866860e95e8ebda65b63ceffc18325379d5a9af98532ef8d3728a8aa5f"
}

REQUIRED_ARTIFACTS: List[str] = [
    "cyclone_track_lat_24h_v3.joblib",
    "cyclone_track_lon_24h_v3.joblib",
    "cyclone_intensity_24h_v3.joblib",
    "cyclone_ri_classifier_24h_v3.joblib"
]

FLOOD_ARTIFACTS: List[str] = [
    "flood_discharge_rf_v3.joblib"
]

LANDSLIDE_ARTIFACTS: List[str] = [
    "landslide_calibrated_rf_v3.joblib"
]

TSUNAMI_ARTIFACTS: List[str] = [
    "tsunami_classifier_rf_v3.joblib",
    "tsunami_scaler_v3.joblib"
]


def compute_sha256(file_path: Path) -> str:
    """Calculates SHA256 hex digest of a file in 64KB blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ------------------------------------------------------------------------------
# MODEL BUNDLES
# ------------------------------------------------------------------------------
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


class FloodModelBundle:
    """In-memory bundle holding verified Flood discharge Random Forest model."""

    def __init__(
        self,
        model: Any,
        source: str,
        load_time_seconds: float,
        checksums: Dict[str, str]
    ):
        self.model = model
        self.source = source
        self.load_time_seconds = load_time_seconds
        self.checksums = checksums

    @property
    def is_ready(self) -> bool:
        return self.model is not None


class LandslideModelBundle:
    """In-memory bundle holding verified Landslide Calibrated Random Forest classifier."""

    def __init__(
        self,
        model: Any,
        source: str,
        load_time_seconds: float,
        checksums: Dict[str, str]
    ):
        self.model = model
        self.source = source
        self.load_time_seconds = load_time_seconds
        self.checksums = checksums

    @property
    def is_ready(self) -> bool:
        return self.model is not None


class TsunamiModelBundle:
    """In-memory bundle holding verified Tsunami Random Forest classifier and feature scaler."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        source: str,
        load_time_seconds: float,
        checksums: Dict[str, str]
    ):
        self.model = model
        self.scaler = scaler
        self.source = source
        self.load_time_seconds = load_time_seconds
        self.checksums = checksums

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None


# ------------------------------------------------------------------------------
# CENTRAL MODEL LOADER
# ------------------------------------------------------------------------------
class ModelLoader:
    """Manages local verification and in-memory loading of all four hazard models."""

    def __init__(self):
        self._cyclone_bundle: Optional[CycloneModelBundle] = None
        self._flood_bundle: Optional[FloodModelBundle] = None
        self._landslide_bundle: Optional[LandslideModelBundle] = None
        self._tsunami_bundle: Optional[TsunamiModelBundle] = None

    def _verify_artifacts(self, model_dir: Path, artifacts: List[str]) -> Dict[str, Path]:
        """Checks artifact existence and verifies cryptographic SHA256 checksums."""
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory '{model_dir}' does not exist. "
                "Ensure model artifacts are committed and available at the configured path."
            )

        paths: Dict[str, Path] = {}
        for filename in artifacts:
            path = model_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Fatal: Required model artifact '{filename}' not found in '{model_dir}'."
                )
            paths[filename] = path

            digest = compute_sha256(path)
            expected = EXPECTED_CHECKSUMS.get(filename)
            if expected and digest != expected:
                raise ValueError(
                    f"FATAL: Cryptographic checksum mismatch for artifact '{filename}'. "
                    f"Actual SHA256: {digest} | Expected: {expected}. "
                    "Aborting startup to prevent corrupted model inference."
                )
            logger.info(f"Verified {filename}: SHA256={digest[:16]}... (MATCH)")

        return paths

    # 1. Cyclone Model Loading (Production)
    def load_cyclone_models(self) -> CycloneModelBundle:
        """Loads and verifies the 4 production cyclone models directly from local repository."""
        if self._cyclone_bundle is not None:
            return self._cyclone_bundle

        start_time = time.time()
        model_dir = settings.resolved_cyclone_model_dir
        logger.info(f"Loading Cyclone model artifacts from: {model_dir}")

        artifact_paths = self._verify_artifacts(model_dir, REQUIRED_ARTIFACTS)

        model_lat = joblib.load(artifact_paths["cyclone_track_lat_24h_v3.joblib"])
        model_lon = joblib.load(artifact_paths["cyclone_track_lon_24h_v3.joblib"])
        model_wind = joblib.load(artifact_paths["cyclone_intensity_24h_v3.joblib"])
        model_ri = joblib.load(artifact_paths["cyclone_ri_classifier_24h_v3.joblib"])

        load_time = round(time.time() - start_time, 3)
        source = f"local_repository ({model_dir})"
        checksums = {f: compute_sha256(p) for f, p in artifact_paths.items()}
        logger.info(f"Successfully loaded 4 Cyclone models in {load_time}s from {source}.")

        self._cyclone_bundle = CycloneModelBundle(
            model_lat=model_lat,
            model_lon=model_lon,
            model_wind=model_wind,
            model_ri=model_ri,
            source=source,
            load_time_seconds=load_time,
            checksums=checksums
        )
        return self._cyclone_bundle

    # 2. Flood Model Loading (Demonstration)
    def load_flood_model(self) -> FloodModelBundle:
        """Loads and verifies Flood river discharge model for demonstration."""
        if self._flood_bundle is not None:
            return self._flood_bundle

        start_time = time.time()
        model_dir = settings.resolved_flood_model_dir
        logger.info(f"Loading Flood model artifacts from: {model_dir}")

        artifact_paths = self._verify_artifacts(model_dir, FLOOD_ARTIFACTS)
        model = joblib.load(artifact_paths["flood_discharge_rf_v3.joblib"])

        load_time = round(time.time() - start_time, 3)
        source = f"local_repository ({model_dir})"
        checksums = {f: compute_sha256(p) for f, p in artifact_paths.items()}
        logger.info(f"Successfully loaded Flood model in {load_time}s from {source}.")

        self._flood_bundle = FloodModelBundle(
            model=model,
            source=source,
            load_time_seconds=load_time,
            checksums=checksums
        )
        return self._flood_bundle

    # 3. Landslide Model Loading (Demonstration)
    def load_landslide_model(self) -> LandslideModelBundle:
        """Loads and verifies Landslide susceptibility model for demonstration."""
        if self._landslide_bundle is not None:
            return self._landslide_bundle

        start_time = time.time()
        model_dir = settings.resolved_landslide_model_dir
        logger.info(f"Loading Landslide model artifacts from: {model_dir}")

        artifact_paths = self._verify_artifacts(model_dir, LANDSLIDE_ARTIFACTS)
        model = joblib.load(artifact_paths["landslide_calibrated_rf_v3.joblib"])

        load_time = round(time.time() - start_time, 3)
        source = f"local_repository ({model_dir})"
        checksums = {f: compute_sha256(p) for f, p in artifact_paths.items()}
        logger.info(f"Successfully loaded Landslide model in {load_time}s from {source}.")

        self._landslide_bundle = LandslideModelBundle(
            model=model,
            source=source,
            load_time_seconds=load_time,
            checksums=checksums
        )
        return self._landslide_bundle

    # 4. Tsunami Model Loading (Demonstration)
    def load_tsunami_model(self) -> TsunamiModelBundle:
        """Loads and verifies Tsunami screening model and scaler for demonstration."""
        if self._tsunami_bundle is not None:
            return self._tsunami_bundle

        start_time = time.time()
        model_dir = settings.resolved_tsunami_model_dir
        logger.info(f"Loading Tsunami model artifacts from: {model_dir}")

        artifact_paths = self._verify_artifacts(model_dir, TSUNAMI_ARTIFACTS)
        model = joblib.load(artifact_paths["tsunami_classifier_rf_v3.joblib"])
        scaler = joblib.load(artifact_paths["tsunami_scaler_v3.joblib"])

        load_time = round(time.time() - start_time, 3)
        source = f"local_repository ({model_dir})"
        checksums = {f: compute_sha256(p) for f, p in artifact_paths.items()}
        logger.info(f"Successfully loaded Tsunami model in {load_time}s from {source}.")

        self._tsunami_bundle = TsunamiModelBundle(
            model=model,
            scaler=scaler,
            source=source,
            load_time_seconds=load_time,
            checksums=checksums
        )
        return self._tsunami_bundle

    def load_all_models(self) -> Dict[str, Any]:
        """Loads all four models (Cyclone for production, Flood/Landslide/Tsunami for demo)."""
        logger.info("Initializing multi-hazard model loader for all four hazard modules...")
        cyclone = self.load_cyclone_models()

        flood = None
        landslide = None
        tsunami = None

        if settings.DEMO_MODE:
            try:
                flood = self.load_flood_model()
            except Exception as e:
                logger.warning(f"Could not initialize Flood demonstration model: {e}")

            try:
                landslide = self.load_landslide_model()
            except Exception as e:
                logger.warning(f"Could not initialize Landslide demonstration model: {e}")

            try:
                tsunami = self.load_tsunami_model()
            except Exception as e:
                logger.warning(f"Could not initialize Tsunami demonstration model: {e}")

        return {
            "cyclone": cyclone,
            "flood": flood,
            "landslide": landslide,
            "tsunami": tsunami
        }

    # Accessors
    def get_bundle(self) -> CycloneModelBundle:
        """Returns the loaded Cyclone bundle (retained for backward compatibility)."""
        if self._cyclone_bundle is None:
            return self.load_cyclone_models()
        return self._cyclone_bundle

    def get_cyclone_bundle(self) -> CycloneModelBundle:
        return self.get_bundle()

    def get_flood_bundle(self) -> FloodModelBundle:
        if self._flood_bundle is None:
            return self.load_flood_model()
        return self._flood_bundle

    def get_landslide_bundle(self) -> LandslideModelBundle:
        if self._landslide_bundle is None:
            return self.load_landslide_model()
        return self._landslide_bundle

    def get_tsunami_bundle(self) -> TsunamiModelBundle:
        if self._tsunami_bundle is None:
            return self.load_tsunami_model()
        return self._tsunami_bundle


# Global singleton instance
model_loader = ModelLoader()

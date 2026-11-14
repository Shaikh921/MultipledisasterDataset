"""
AAPADAMITRA API Configuration
Centralized settings management supporting Render free-tier environment variables
and local repository model deployment.
"""

from typing import List, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Security
    API_KEY: str = "aapadamitra-dev-key"
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Local Model Storage (Committed directly to repository)
    MODEL_DIR: str = "models/training/cyclone/artifacts"
    CYCLONE_MODEL_DIR: str = "models/training/cyclone/artifacts"
    FLOOD_MODEL_DIR: str = "models/training/flood/artifacts"
    LANDSLIDE_MODEL_DIR: str = "models/training/landslide/artifacts"
    TSUNAMI_MODEL_DIR: str = "models/training/tsunami/artifacts"

    # Multi-Hazard Demonstration Mode
    DEMO_MODE: bool = True

    # Operational Limits & Constraints (Render Free Tier: 512 MB)
    MAX_MEMORY_RSS_MB: float = 512.0
    PROMOTED_HAZARDS: List[str] = ["cyclone"]

    # Pydantic v2 Settings Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def resolved_model_dir(self) -> Path:
        """Resolves local cyclone model artifact directory (retained for backward compatibility)."""
        return self.resolved_cyclone_model_dir

    @property
    def resolved_cyclone_model_dir(self) -> Path:
        """Resolves local cyclone model artifact directory."""
        configured = Path(self.MODEL_DIR)
        if configured.exists() and configured.is_dir():
            return configured

        alt_configured = Path(self.CYCLONE_MODEL_DIR)
        if alt_configured.exists() and alt_configured.is_dir():
            return alt_configured

        # Fallback paths within the repository
        candidates = [
            Path("models/training/cyclone/artifacts"),
            Path("deployment/models/training/cyclone/artifacts"),
            Path("improved_cyclone_model")
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        return configured

    @property
    def resolved_flood_model_dir(self) -> Path:
        """Resolves local flood model artifact directory."""
        configured = Path(self.FLOOD_MODEL_DIR)
        if configured.exists() and configured.is_dir():
            return configured
        candidates = [
            Path("models/training/flood/artifacts"),
            Path("deployment/models/training/flood/artifacts"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return configured

    @property
    def resolved_landslide_model_dir(self) -> Path:
        """Resolves local landslide model artifact directory."""
        configured = Path(self.LANDSLIDE_MODEL_DIR)
        if configured.exists() and configured.is_dir():
            return configured
        candidates = [
            Path("models/training/landslide/artifacts"),
            Path("deployment/models/training/landslide/artifacts"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return configured

    @property
    def resolved_tsunami_model_dir(self) -> Path:
        """Resolves local tsunami model artifact directory."""
        configured = Path(self.TSUNAMI_MODEL_DIR)
        if configured.exists() and configured.is_dir():
            return configured
        candidates = [
            Path("models/training/tsunami/artifacts"),
            Path("deployment/models/training/tsunami/artifacts"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return configured


settings = Settings()

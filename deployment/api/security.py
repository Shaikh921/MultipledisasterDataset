"""
AAPADAMITRA API Security
Constant-time API Key verification for remote inference protection.
Enforces secure production keys and rejects default development keys in production.
"""

import secrets
import logging
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from deployment.api.config import settings

logger = logging.getLogger("API_Security")

# Header definition: X-API-Key
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validates the provided API key using constant-time string comparison.
    Raises HTTP 401 Unauthorized if the key is missing or invalid.
    Raises HTTP 500 if the server is in production mode but using a development key.
    """
    expected_key = settings.API_KEY

    # In production, reject default development key
    if settings.ENVIRONMENT.lower() == "production" and expected_key in ["aapadamitra-dev-key", "", None]:
        logger.critical("Security Violation: Development API key used in production environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: A unique production API key must be set in the environment."
        )

    # If no key provided
    if not api_key:
        logger.warning("Unauthorized request: Missing API key in header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a valid key in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key.strip(), expected_key.strip()):
        logger.warning("Unauthorized request: Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Provide a valid key in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return api_key

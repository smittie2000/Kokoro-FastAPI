"""
Two-tier API key authentication for TTS service.

Tier 1 (API keys): Protects /v1/* and /web/* routes.
    - Validates Authorization: Bearer <key> against API_KEYS env var.
    - Empty API_KEYS = auth disabled (backwards-compatible for local dev).

Tier 2 (Debug key): Protects /debug/* and /dev/* routes.
    - Validates Authorization: Bearer <key> against DEBUG_API_KEY env var.
    - Empty DEBUG_API_KEY = endpoints return 403 (disabled entirely).
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

# auto_error=False so missing header returns None instead of 403
# (we handle the error ourselves to distinguish "no key configured" from "bad key")
_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Validate Bearer token against API_KEYS. No-op if API_KEYS is empty."""
    allowed = settings.api_key_set
    if not allowed:
        return  # Auth disabled — no keys configured

    if not credentials or credentials.credentials not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_debug_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Validate Bearer token against DEBUG_API_KEY. 403 if DEBUG_API_KEY is empty."""
    if not settings.debug_api_key:
        raise HTTPException(status_code=403, detail="Debug endpoints are disabled")

    if not credentials or credentials.credentials != settings.debug_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing debug key")

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status


API_KEY_HEADER = "X-API-Key"


def get_expected_api_key() -> str:
    return os.getenv("API_KEY", "dev-api-key")


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> str:
    expected = get_expected_api_key()

    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    return x_api_key
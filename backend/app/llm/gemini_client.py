"""Timeout-bounded structured-output access to the Gemini API."""

from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 10


class GeminiError(RuntimeError):
    """Base exception for a Gemini integration failure."""


class GeminiUnavailableError(GeminiError):
    """Raised when Gemini cannot be configured in the current environment."""


class GeminiRequestError(GeminiError):
    """Raised when a bounded Gemini request fails."""


class GeminiResponseError(GeminiError):
    """Raised when Gemini does not return a valid JSON object."""


def _load_environment() -> None:
    """Load a local .env file when python-dotenv is available."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _configured_timeout(timeout_seconds: int | None) -> int:
    """Resolve an explicit timeout or the optional environment configuration."""

    if timeout_seconds is not None:
        return timeout_seconds

    configured_timeout = os.getenv("GEMINI_TIMEOUT_SECONDS")
    if not configured_timeout:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        return int(configured_timeout)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def request_structured_output(
    prompt: str,
    response_schema: dict[str, Any],
    *,
    timeout_seconds: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Request a JSON object from Gemini within a fixed client-side timeout.

    The Gemini SDK receives its timeout in milliseconds. Callers handle these
    bounded failures with deterministic fallbacks, so no LLM failure can stop
    the analytical pipeline.
    """

    resolved_timeout_seconds = _configured_timeout(timeout_seconds)
    if resolved_timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    _load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailableError("GEMINI_API_KEY is not configured")

    try:
        from google import genai
    except ImportError as error:
        raise GeminiUnavailableError(
            "google-genai is not installed; install backend/requirements.txt"
        ) from error

    client: Any | None = None
    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": resolved_timeout_seconds * 1000},
        )
        response = client.models.generate_content(
            model=model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )
        response_text = getattr(response, "text", None)
    except Exception as error:
        raise GeminiRequestError(f"Gemini request failed: {type(error).__name__}") from error
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    if not isinstance(response_text, str) or not response_text.strip():
        raise GeminiResponseError("Gemini returned an empty response")

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise GeminiResponseError("Gemini returned malformed JSON") from error

    if not isinstance(payload, dict):
        raise GeminiResponseError("Gemini response must be a JSON object")

    return payload

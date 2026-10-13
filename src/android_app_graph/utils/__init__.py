"""Android-App-Graph utilities."""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx
from openrouter import OpenRouter

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PAGE_DETAIL_MODEL = "openai/gpt-6.1-sol"
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_INSTRUCTION_MODEL = "openai/gpt-6.1-sol"
DEFAULT_ACTION_MODEL = "qwen/qwen3.8-flash"


class ConfiguredOpenRouter(OpenRouter):
    """Keep optional chat parameters on the client belonging to each role."""

    temperature: float | None = None


def validate_embedding_endpoint(base_url: str, *, role: str) -> None:
    """Reject native Google APIs that cannot accept the OpenRouter wire format."""
    host = urlsplit(base_url).hostname or ""
    if (
        host == "generativelanguage.googleapis.com"
        or host == "aiplatform.googleapis.com"
        or host.endswith((".aiplatform.googleapis.com", "-aiplatform.googleapis.com"))
    ):
        msg = (
            f"{role}: native Google embedding endpoints are incompatible with OpenRouter requests. "
            f"Set {role}.base_url to {DEFAULT_BASE_URL}, use OPENROUTER_API_KEY, and select "
            f"{DEFAULT_EMBEDDING_MODEL} (or another OpenRouter embedding model)."
        )
        raise ValueError(msg)


def _trust_environment(base_url: str) -> bool:
    host = urlsplit(base_url).hostname or ""
    if host.rstrip(".").lower() == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def resolve_env(value: object) -> str | None:
    """Resolve ``${ENV_VAR}`` references in a string."""
    if value is None:
        return None
    m = re.fullmatch(r"\$\{(\w+)\}", str(value).strip())
    if m:
        return os.environ.get(m.group(1))
    return str(value)


def make_client(
    cfg: Mapping[str, object] | None,
    *,
    default_model: str = DEFAULT_PAGE_DETAIL_MODEL,
    http_client: httpx.Client | None = None,
    timeout: float | None = None,
    role: str = "chat",
) -> tuple[OpenRouter, str]:
    """Create the official SDK client with per-role overrides and a shared key.

    Timeouts are configured in seconds here and milliseconds in the SDK.
    The application owns bounded retries; the SDK's hour-long retry window
    must not multiply those attempts.
    """
    cfg = cfg or {}
    if role == "chat" and default_model == DEFAULT_EMBEDDING_MODEL:
        role = "embedding"
    base_url = resolve_env(cfg.get("base_url")) or DEFAULT_BASE_URL
    if role in {"embedding", "image_embedding"}:
        validate_embedding_endpoint(base_url, role=role)
    api_key = resolve_env(cfg.get("api_key")) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            f"Missing API key for {role}: configure {role}.api_key or set OPENROUTER_API_KEY"
        )
    model = resolve_env(cfg.get("model")) or default_model
    request_timeout = (
        timeout
        if timeout is not None
        else float(resolve_env(cfg.get("request_timeout", cfg.get("timeout", 60))) or "60")
    )
    temperature_value = resolve_env(cfg.get("temperature"))
    temperature = float(temperature_value) if temperature_value is not None else None
    if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError(f"{role}.temperature must be a finite number between 0 and 2")
    client = ConfiguredOpenRouter(
        api_key=api_key,
        server_url=base_url.rstrip("/"),
        client=http_client
        if http_client is not None
        else httpx.Client(trust_env=_trust_environment(base_url)),
        timeout_ms=int(request_timeout * 1000),
        retry_config=None,
    )
    client.temperature = temperature
    return client, model

"""Android-App-Graph utilities."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

import httpx
from openrouter import OpenRouter

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PAGE_DETAIL_MODEL = "openai/gpt-6.1-sol"
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_INSTRUCTION_MODEL = "openai/gpt-6.1-sol"
DEFAULT_ACTION_MODEL = "qwen/qwen3.8-flash"


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
) -> tuple[OpenRouter, str]:
    """Create the official SDK client with per-role overrides and a shared key.

    Timeouts are configured in seconds here and milliseconds in the SDK.
    The application owns bounded retries; the SDK's hour-long retry window
    must not multiply those attempts.
    """
    cfg = cfg or {}
    api_key = resolve_env(cfg.get("api_key")) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing API key: configure api_key or set OPENROUTER_API_KEY")
    base_url = resolve_env(cfg.get("base_url")) or DEFAULT_BASE_URL
    model = resolve_env(cfg.get("model")) or default_model
    request_timeout = (
        timeout
        if timeout is not None
        else float(resolve_env(cfg.get("request_timeout", cfg.get("timeout", 60))) or "60")
    )
    return OpenRouter(
        api_key=api_key,
        server_url=base_url.rstrip("/"),
        client=http_client,
        timeout_ms=int(request_timeout * 1000),
        retry_config=None,
    ), model

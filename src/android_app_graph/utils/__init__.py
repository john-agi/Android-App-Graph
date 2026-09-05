"""Android-App-Graph utilities."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI


def resolve_env(value: str | None) -> str | None:
    """Resolve ``${ENV_VAR}`` references in a string."""
    if value is None:
        return None
    m = re.fullmatch(r"\$\{(\w+)\}", str(value).strip())
    if m:
        return os.environ.get(m.group(1))
    return str(value)


def make_client(cfg: dict[str, Any] | None) -> tuple[OpenAI, str]:
    """Create an OpenAI client from a per-call config block.

    Args:
        cfg: Dict with keys ``model``, ``base_url``, ``api_key`` (all optional).

    Returns:
        (client, model_name)
    """
    cfg = cfg or {}
    api_key = resolve_env(cfg.get("api_key")) or os.environ.get("OPENAI_API_KEY")
    base_url = resolve_env(cfg.get("base_url"))
    model = resolve_env(cfg.get("model")) or "gpt-4o"

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs), model

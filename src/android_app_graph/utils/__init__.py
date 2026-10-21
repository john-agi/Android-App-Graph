"""Client configuration; required settings: README.md#configure-api-credentials."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping
from typing import NamedTuple

import httpx
from openrouter import OpenRouter

from android_app_graph.payloads import as_object_dict

ROLES = ("page_detail", "embedding", "image_embedding", "instruction", "action")
# Earlier releases and AITK controller.yaml files copied from upstream UI-KOBE set these.
REMOVED_KEYS = ("api_key", "base_url", "native_base_url")
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_AUDIT_TIMEOUT = 300.0
MAX_TEMPERATURE = 2


class ModelSettings(NamedTuple):
    """One role's per-request settings, resolved before any device or network work."""

    model: str
    temperature: float | None = None
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


def open_client(api_key: str, *, http_client: httpx.Client | None = None) -> OpenRouter:
    """Create the run's one OpenRouter client, shared by every role.

    Only tests pass ``http_client``; otherwise the SDK creates and owns the HTTP
    client. The SDK's default retry policy is kept as is:
    README.md#requests-and-recovery.
    """
    return OpenRouter(api_key=api_key, client=http_client)


def removed_key_errors(vlm_config: Mapping[str, object]) -> list[str]:
    """Name every removed endpoint or credential key left in a role block."""
    return [
        f"{role}.{key} was removed: OpenRouter is the only endpoint and OPENROUTER_API_KEY "
        "the only credential; delete this key."
        for role in ROLES
        for key in REMOVED_KEYS
        if key in as_object_dict(vlm_config.get(role))
    ]


def resolve_api_key() -> str:
    """Return OPENROUTER_API_KEY, the only credential."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        msg = (
            "OPENROUTER_API_KEY is required: export your OpenRouter API key; it is the only "
            "credential for every role."
        )
        raise ValueError(msg)
    return api_key


def resolve_env(value: object) -> str | None:
    """Resolve ``${ENV_VAR}`` references in a string."""
    if value is None:
        return None
    m = re.fullmatch(r"\$\{(\w+)\}", str(value).strip())
    if m:
        return os.environ.get(m.group(1))
    return str(value)


def resolve_model(value: object, *, role: str) -> str:
    """Require an explicit model, using the role environment variable when omitted."""
    env_role = "embedding" if role == "image_embedding" else role
    variable = f"APP_GRAPH_{env_role.upper()}_MODEL"
    source = f"${{{variable}}}" if value is None else value
    model = resolve_env(source) if isinstance(source, str) else None
    if not model or not model.strip() or "${" in model:
        msg = (
            f"{role}.model is required: set {variable} or configure an explicit model ID "
            f"for {role}; empty values and unresolved environment references are invalid."
        )
        raise ValueError(msg)
    return model.strip()


def _seconds(value: object, *, setting: str, default: float) -> float:
    message = f"{setting} must be a positive finite number of seconds"
    try:
        seconds = float(resolve_env(value) or default)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(message)
    return seconds


def resolve_request_timeout(cfg: Mapping[str, object], *, role: str) -> float:
    """Validate seconds before opening connections or sending requests."""
    return _seconds(
        cfg.get("request_timeout", cfg.get("timeout")),
        setting=f"{role}.request_timeout",
        default=DEFAULT_REQUEST_TIMEOUT,
    )


def resolve_audit_timeout(vlm_config: Mapping[str, object]) -> float:
    """Validate the page_detail timeout for whole-graph audits, whose prompts are long."""
    return _seconds(
        as_object_dict(vlm_config.get("page_detail")).get("audit_request_timeout"),
        setting="page_detail.audit_request_timeout",
        default=DEFAULT_AUDIT_TIMEOUT,
    )


def resolve_temperature(cfg: Mapping[str, object], *, role: str) -> float | None:
    """Validate an optional sampling temperature before any request."""
    message = f"{role}.temperature must be a finite number between 0 and {MAX_TEMPERATURE}"
    value = resolve_env(cfg.get("temperature"))
    if value is None:
        return None
    try:
        temperature = float(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= MAX_TEMPERATURE:
        raise ValueError(message)
    return temperature


def _role_config(vlm_config: Mapping[str, object], role: str) -> dict[str, object]:
    """Return a role's block; image embeddings inherit the embedding model and timeout."""
    cfg = as_object_dict(vlm_config.get(role))
    if role != "image_embedding":
        return cfg
    embedding = as_object_dict(vlm_config.get("embedding"))
    inherited = dict(cfg)
    if "model" not in cfg:
        inherited["model"] = embedding.get("model")
    if "request_timeout" not in cfg and "timeout" not in cfg:
        inherited["request_timeout"] = embedding.get("request_timeout", embedding.get("timeout"))
    return inherited


def _checked[T](resolve: Callable[[], T], errors: list[str]) -> T | None:
    try:
        return resolve()
    except ValueError as exc:
        errors.append(str(exc))
        return None


def _role_settings(
    vlm_config: Mapping[str, object], role: str, errors: list[str]
) -> ModelSettings | None:
    """Resolve one role, recording every invalid field rather than only the first."""
    cfg = _role_config(vlm_config, role)
    model = _checked(lambda: resolve_model(cfg.get("model"), role=role), errors)
    temperature = _checked(lambda: resolve_temperature(cfg, role=role), errors)
    timeout = _checked(lambda: resolve_request_timeout(cfg, role=role), errors)
    if model is None or timeout is None:
        return None
    return ModelSettings(model, temperature, timeout)


def resolve_model_settings(
    vlm_config: Mapping[str, object], *, roles: tuple[str, ...]
) -> dict[str, ModelSettings]:
    """Validate an operation's settings, and removed keys in every role block, all at once."""
    errors = removed_key_errors(vlm_config)
    settings: dict[str, ModelSettings] = {}
    for role in roles:
        resolved = _role_settings(vlm_config, role, errors)
        if resolved is not None:
            settings[role] = resolved
    if "page_detail" in roles:
        _checked(lambda: resolve_audit_timeout(vlm_config), errors)
    if errors:
        raise ValueError(
            "Invalid vlm configuration: " + "; ".join(error.rstrip(".") for error in errors)
        )
    return settings

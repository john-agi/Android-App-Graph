"""Client configuration; required settings: README.md#configure-api-credentials."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Self, override

import httpx
from openrouter import OpenRouter

from android_app_graph.payloads import as_object_dict

ROLES = ("page_detail", "embedding", "image_embedding", "instruction", "action")
# Earlier releases and AITK controller.yaml files copied from upstream UI-KOBE set these.
REMOVED_KEYS = ("api_key", "base_url", "native_base_url")


class ConfiguredOpenRouter(OpenRouter):
    """Keep role settings and explicitly own the SDK-created HTTP client."""

    temperature: float | None = None
    owns_http_client: bool = False
    _scope_depth: int = 0

    @override
    def __enter__(self) -> Self:
        client = self.sdk_configuration.client
        if self.owns_http_client and isinstance(client, httpx.Client) and client.is_closed:
            self.sdk_configuration.client = httpx.Client(follow_redirects=True)
        self._scope_depth += 1
        return self

    @override
    def __exit__(self, *_args: object, **_kwargs: object) -> None:
        self._scope_depth -= 1
        if (
            self._scope_depth == 0
            and self.owns_http_client
            and self.sdk_configuration.client is not None
        ):
            self.sdk_configuration.client.close()


@contextmanager
def client_scope(client: OpenRouter) -> Iterator[OpenRouter]:
    """Reuse an active application scope; leave externally supplied SDK clients alone."""
    if isinstance(client, ConfiguredOpenRouter):
        with client:
            yield client
    else:
        yield client


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
        raise ValueError(
            "OPENROUTER_API_KEY is required: export your OpenRouter API key; it is the only "
            "credential for every role."
        )
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
        raise ValueError(
            f"{role}.model is required: set {variable} or configure an explicit model ID "
            f"for {role}; empty values and unresolved environment references are invalid."
        )
    return model.strip()


def resolve_model_roles(
    vlm_config: Mapping[str, object], *, roles: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    """Resolve the operation's models together before constructing clients or devices."""
    resolved: dict[str, dict[str, object]] = {}
    errors = removed_key_errors(vlm_config)
    for role in roles:
        cfg = as_object_dict(vlm_config.get(role)).copy()
        value = cfg.get("model")
        if role == "image_embedding" and "model" not in cfg:
            value = as_object_dict(vlm_config.get("embedding")).get("model")
        try:
            cfg["model"] = resolve_model(value, role=role)
        except ValueError as exc:
            errors.append(str(exc))
        resolved[role] = cfg
    if errors:
        raise ValueError("Invalid vlm configuration:\n" + "\n".join(errors))
    return resolved


def resolve_request_timeout(cfg: Mapping[str, object], *, role: str) -> float:
    """Validate seconds before opening connections or sending requests."""
    message = f"{role}.request_timeout must be a positive finite number of seconds"
    try:
        value = float(resolve_env(cfg.get("request_timeout", cfg.get("timeout"))) or "60")
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(message)
    return value


def resolve_temperature(cfg: Mapping[str, object], *, role: str) -> float | None:
    """Validate an optional sampling temperature before creating the client."""
    message = f"{role}.temperature must be a finite number between 0 and 2"
    value = resolve_env(cfg.get("temperature"))
    if value is None:
        return None
    try:
        temperature = float(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError(message)
    return temperature


def make_client(
    cfg: Mapping[str, object] | None,
    *,
    role: str,
    http_client: httpx.Client | None = None,
    timeout: float | None = None,
) -> tuple[OpenRouter, str]:
    """Create the official SDK client with per-role overrides and the shared key.

    Timeouts are configured in seconds here and milliseconds in the SDK.
    The application owns bounded retries; the SDK's hour-long retry window
    must not multiply those attempts. Only tests pass ``http_client``; otherwise
    the SDK creates its own.
    """
    cfg = cfg or {}
    model = resolve_model(cfg.get("model"), role=role)
    request_timeout = resolve_request_timeout(
        {"request_timeout": timeout} if timeout is not None else cfg, role=role
    )
    temperature = resolve_temperature(cfg, role=role)
    client = ConfiguredOpenRouter(
        api_key=resolve_api_key(),
        client=http_client,
        timeout_ms=int(request_timeout * 1000),
        retry_config=None,
    )
    client.owns_http_client = http_client is None
    client.temperature = temperature
    return client, model

"""Android-App-Graph utilities."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
from collections.abc import Mapping
from typing import Self, override
from urllib.parse import urlsplit

import httpx
from openrouter import OpenRouter
from openrouter._hooks.sdkhooks import SDKHooks
from openrouter._hooks.types import AfterSuccessContext, AfterSuccessHook

from android_app_graph.payloads import as_object_dict

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PAGE_DETAIL_MODEL = "openai/gpt-6.1-sol"
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_INSTRUCTION_MODEL = "openai/gpt-6.1-sol"
DEFAULT_ACTION_MODEL = "qwen/qwen3.8-flash"


class ConfiguredOpenRouter(OpenRouter):
    """Keep role settings and explicitly own application-created HTTP clients."""

    temperature: float | None = None
    owns_http_client: bool = False
    trust_env: bool = True

    @override
    def __enter__(self) -> Self:
        client = self.sdk_configuration.client
        if self.owns_http_client and isinstance(client, httpx.Client) and client.is_closed:
            self.sdk_configuration.client = httpx.Client(
                trust_env=self.trust_env, follow_redirects=True
            )
        return self

    @override
    def __exit__(self, *_args: object, **_kwargs: object) -> None:
        if self.owns_http_client and self.sdk_configuration.client is not None:
            self.sdk_configuration.client.close()


class _ChatFingerprintCompatibility(AfterSuccessHook):
    @override
    def after_success(
        self, hook_ctx: AfterSuccessContext, response: httpx.Response
    ) -> httpx.Response:
        if (
            hook_ctx.operation_id != "sendChatCompletionRequest"
            or not response.is_success
            or "application/json" not in response.headers.get("content-type", "")
        ):
            return response
        try:
            payload = as_object_dict(response.json())
            request_payload = as_object_dict(json.loads(response.request.content))
        except ValueError, UnicodeDecodeError:
            return response
        if (
            payload.get("object") != "chat.completion"
            or "error" in payload
            or "system_fingerprint" in payload
            or request_payload.get("stream") is True
        ):
            return response
        # SDK 1.3.43 requires this nullable field; compatible servers may omit it.
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        adjusted = httpx.Response(
            response.status_code,
            json={**payload, "system_fingerprint": None},
            headers=headers,
            request=response.request,
            extensions=response.extensions,
        )
        response.close()
        return adjusted


def validate_embedding_config(cfg: Mapping[str, object], *, role: str) -> None:
    """Reject obsolete endpoint fields before resolving or sending credentials."""
    if "native_base_url" in cfg:
        raise ValueError(
            f"{role}.native_base_url is unsupported. Set {role}.base_url to "
            f"{DEFAULT_BASE_URL} (or a compatible endpoint), configure {role}.api_key "
            f"with OPENROUTER_API_KEY (or that endpoint's credential), and select "
            f"{DEFAULT_EMBEDDING_MODEL} (or a model supported by that endpoint)."
        )


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


def resolve_endpoint_base_url(
    value: object, *, default: str = DEFAULT_BASE_URL, api_key: object = None
) -> str:
    """Do not redirect an explicit credential when its endpoint variable is missing."""
    base_url = resolve_env(value)
    if value and not base_url and resolve_env(api_key):
        raise ValueError(
            "Cannot resolve endpoint for an explicit API key; set the endpoint variable "
            "before using this credential."
        )
    return base_url or default


def _endpoint_origin(base_url: str) -> tuple[str, str, int]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Model endpoints must be absolute HTTP or HTTPS URLs")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname, port


def resolve_endpoint_api_key(base_url: str, *credentials: tuple[str, object]) -> str | None:
    """Inherit a credential only within its configured URL origin."""
    origin = _endpoint_origin(base_url)
    for endpoint, value in (*credentials, (DEFAULT_BASE_URL, "${OPENROUTER_API_KEY}")):
        key = resolve_env(value)
        if key and _endpoint_origin(endpoint) == origin:
            return key
    return None


def make_client(
    cfg: Mapping[str, object] | None,
    *,
    role: str,
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
    base_url = resolve_endpoint_base_url(cfg.get("base_url"), api_key=cfg.get("api_key"))
    if role in {"embedding", "image_embedding"}:
        validate_embedding_config(cfg, role=role)
        validate_embedding_endpoint(base_url, role=role)
    api_key = resolve_endpoint_api_key(base_url, (base_url, cfg.get("api_key")))
    if not api_key:
        raise ValueError(
            f"Missing API key for {role}: configure {role}.api_key for this endpoint. "
            "OPENROUTER_API_KEY is inherited only by the HTTPS OpenRouter origin."
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
    trust_env = _trust_environment(base_url)
    transport_client = (
        http_client
        if http_client is not None
        else httpx.Client(trust_env=trust_env, follow_redirects=True)
    )
    try:
        client = ConfiguredOpenRouter(
            api_key=api_key,
            server_url=base_url.rstrip("/"),
            client=transport_client,
            timeout_ms=int(request_timeout * 1000),
            retry_config=None,
        )
        hooks: object = client.sdk_configuration.__dict__["_hooks"]
        if not isinstance(hooks, SDKHooks):
            raise TypeError("Expected OpenRouter SDK hooks")
        hooks.register_after_success_hook(_ChatFingerprintCompatibility())
    except BaseException:
        if http_client is None:
            transport_client.close()
        raise
    client.owns_http_client = http_client is None
    client.trust_env = trust_env
    client.temperature = temperature
    return client, model

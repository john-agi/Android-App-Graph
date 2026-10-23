"""The config keys the commands accept, and each role's client settings.

Required settings: README.md#configure-api-credentials.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import NamedTuple

import httpx
from openrouter import OpenRouter

from android_app_graph.payloads import as_object_dict
from android_app_graph.retrying import DeadlineTransport

ROLES = ("page_detail", "embedding", "image_embedding", "instruction", "action")
_CHAT_ROLES = ("page_detail", "instruction", "action")


class MatchingSettings(NamedTuple):
    """How screen matching places a screenshot: how many nodes each short-list source
    proposes, and how many of them the verifier checks; README.md#screen-matching."""

    verifier_calls: int = 3
    image_candidates: int = 3
    text_candidates: int = 2


DEFAULT_MATCHING = MatchingSettings()

# Every key app-graph or app-graph-audit uses and no other, by path; "[]" stands for each
# entry of a list. Both commands accept all of them, since they share one config file.
CONFIG_KEYS = frozenset(
    {
        "device.udid",
        "device.avd_name",
        "experiment.graph_dir",
        "experiment.max_steps",
        "experiment.resume_from",
        "experiment.coverage_checkpoint_steps",
        "experiment.coverage_checkpoint_top_k",
        "apps[].name",
        "apps[].package_name",
        "apps[].launch_activity",
        "vlm.page_detail.audit_request_timeout",
        *(f"vlm.{role}.{key}" for role in ROLES for key in ("model", "request_timeout", "timeout")),
        *(f"vlm.{role}.temperature" for role in _CHAT_ROLES),
        *(f"vlm.matching.{setting}" for setting in MatchingSettings._fields),
    }
)
_SECTIONS = frozenset(
    key[: separator.start()] for key in CONFIG_KEYS for separator in re.finditer(r"\.|\[\]", key)
)
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_AUDIT_TIMEOUT = 300.0
MAX_TEMPERATURE = 2


class ModelSettings(NamedTuple):
    """One role's per-request settings, resolved before any device or network work."""

    model: str
    temperature: float | None = None
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


@contextmanager
def open_client(api_key: str, *, http_client: httpx.Client | None = None) -> Generator[OpenRouter]:
    """Open the run's one OpenRouter client, shared by every role, and close it on exit.

    Its HTTP client ends each attempt at the request's timeout by the wall clock:
    README.md#requests-and-recovery. The SDK never closes a client it is given,
    so this closes the one it builds. Only tests pass ``http_client``, which
    stays theirs to close.
    """
    with ExitStack() as opened:
        if http_client is None:
            http_client = opened.enter_context(
                httpx.Client(
                    follow_redirects=True, transport=DeadlineTransport(httpx.HTTPTransport())
                )
            )
        client = OpenRouter(api_key=api_key, client=http_client)
        opened.enter_context(client)
        yield client


def _child(parent: str, key: object) -> str:
    return f"{parent}.{key}" if parent else str(key)


def _unknown_keys(value: object, pattern: str, path: str) -> Iterator[str]:
    """Yield the path of each key under ``value`` that is not in ``CONFIG_KEYS``.

    ``pattern`` is ``path`` with ``[]`` for each list index. The value of an
    accepted key is not searched; a section of the wrong shape names what it holds.
    """
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _unknown_keys(item, f"{pattern}[]", f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            key_pattern, key_path = _child(pattern, key), _child(path, key)
            if key_pattern in _SECTIONS:
                yield from _unknown_keys(item, key_pattern, key_path)
            elif key_pattern not in CONFIG_KEYS:
                yield key_path


def check_config_keys(config: object) -> None:
    """Reject a loaded config holding any key no command uses, naming every such key by
    its path and never its value, which can be a credential."""
    unknown = list(_unknown_keys(config, "", ""))
    if unknown:
        msg = f"Unknown configuration keys: {', '.join(unknown)}"
        raise ValueError(msg)


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
    """Require an explicit model, using the role environment variable when omitted.

    Image embeddings read ``APP_GRAPH_EMBEDDING_MODEL``, so one variable chooses the
    model for text and screenshot embeddings.
    """
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
    """Validate an operation's settings all at once."""
    errors: list[str] = []
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


def resolve_matching_settings(vlm_config: Mapping[str, object]) -> MatchingSettings:
    """Validate ``vlm.matching`` all at once: each setting a whole number of 0 or more,
    and one it leaves out taking its value in ``DEFAULT_MATCHING``."""
    matching = as_object_dict(vlm_config.get("matching"))
    counts: dict[str, int] = {}
    errors: list[str] = []
    for setting, default in DEFAULT_MATCHING._asdict().items():
        value = matching.get(setting, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"vlm.matching.{setting} must be a whole number of 0 or more")
        else:
            counts[setting] = value
    if errors:
        raise ValueError("Invalid vlm configuration: " + "; ".join(errors))
    return MatchingSettings(**counts)

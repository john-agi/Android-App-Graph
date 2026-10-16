"""Offline regressions for the OpenRouter migration's recovery and configuration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError

from android_app_graph import embedding_cache
from android_app_graph.adapters import aitk_translator
from android_app_graph.embedding_cache import resolve_image_embedding_settings
from android_app_graph.utils import make_client, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager

_CLICK = '{"action":"click","coordinate":[500,500]}'


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 408, 409, 425, 429, 500, 503])
def test_http_status_controls_attempts_and_backoff(
    status: int,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_responses.extend(
        httpx.Response(status, json={"error": {"code": status, "message": "failed"}})
        for _ in range(3)
    )
    monkeypatch.setattr(
        embedding_cache,
        "get_openrouter_image_embedding",
        lambda *_args, **_kwargs: vlm_utils.get_image_embedding(
            router_client, png_screenshot, model="google/gemini-embedding-2"
        ),
    )
    with pytest.raises(OpenRouterError):
        embedding_cache.compute_embedding_with_retry(
            "test",
            png_screenshot,
            model="test",
            base_url="https://openrouter.ai/api/v1",
            app_name="demo",
            node_id="home",
        )
    transient = status in {408, 409, 425, 429} or status >= 500
    assert len(router_requests) == (3 if transient else 1)
    assert no_sleep == ([2.0, 4.0] if transient else [])


@pytest.mark.parametrize(
    ("endpoint", "trust_env"),
    [
        ("https://openrouter.ai/api/v1", True),
        ("https://router.example/v1", True),
        ("http://localhost:8000/v1", False),
        ("http://127.0.0.1:8000/v1", False),
        ("http://[::1]:8000/v1", False),
    ],
)
def test_v2_endpoint_proxy_policy(
    endpoint: str, trust_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    client, _ = make_client(
        {"model": "qwen/qwen3.8-flash", "api_key": "test", "base_url": endpoint}, role="action"
    )
    assert isinstance(client.sdk_configuration.client, httpx.Client)
    assert client.sdk_configuration.client.trust_env is trust_env
    client.sdk_configuration.client.close()


@pytest.mark.usefixtures("model_environment")
def test_temperatures_are_independent_across_roles(
    tmp_path: Path,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    png_screenshot: str,
    monkeypatch: pytest.MonkeyPatch,
    chat_response: Callable[..., httpx.Response],
) -> None:
    def factory(cfg: dict[str, object] | None, *, role: str) -> tuple[OpenRouter, str]:
        return make_client(cfg, role=role, http_client=router_http)

    monkeypatch.setattr(aitk_translator, "make_client", factory)
    translator = aitk_translator.UIKobeV2Translator(
        graph_dir=str(tmp_path),
        vlm_config={
            "action": {"api_key": "test", "temperature": 0.0},
            "page_detail": {"api_key": "test", "temperature": 0.4},
            "instruction": {"api_key": "test", "temperature": 0.8},
        },
    )
    router_responses.extend(
        [
            chat_response(content='{"action":"wait"}'),
            chat_response(content='{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=chat_body),
        ]
    )
    vlm_utils.predict_next_action(
        translator.model_client, png_screenshot, "wait", model=translator.model_name
    )
    vlm_utils.describe_page_and_state(
        translator.desc_client, png_screenshot, model=translator.desc_model
    )
    translator._ask_with_screenshot("plan", None, str, "test")
    assert [json.loads(request.content)["temperature"] for request in router_requests] == [
        0.0,
        0.4,
        0.8,
    ]


@pytest.mark.usefixtures("model_environment")
def test_instruction_credentials_error_names_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"instruction.*OPENROUTER_API_KEY"):
        aitk_translator.UIKobeV2Translator(
            graph_dir=str(tmp_path),
            vlm_config={
                "action": {"api_key": "test"},
                "page_detail": {"api_key": "test"},
            },
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://generativelanguage.googleapis.com/v1beta",
        "https://us-central1-aiplatform.googleapis.com/v1",
    ],
)
def test_native_google_embedding_config_rejected_before_requests(endpoint: str) -> None:
    with pytest.raises(ValueError, match=r"embedding.*OpenRouter.*OPENROUTER_API_KEY"):
        make_client(
            {"model": "google/gemini-embedding-2", "api_key": "test", "base_url": endpoint},
            role="embedding",
        )
    with pytest.raises(ValueError, match=r"image_embedding.*OpenRouter"):
        resolve_image_embedding_settings({"image_embedding": {"base_url": endpoint}})


@pytest.mark.parametrize("storage", ["inline", "sidecar"])
@pytest.mark.parametrize("model", [None, "provider/old-same-dimensions"])
@pytest.mark.parametrize("operation", ["find", "identify"])
def test_text_embedding_unknown_or_mismatched_provenance_blocks_comparison(
    storage: str,
    model: str | None,
    operation: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    path = tmp_path / "demo.json"
    node: dict[str, object] = {
        "id": "home",
        "activity": "com.demo.Home",
        "page_description": "home",
    }
    if storage == "inline":
        node["description_embedding"] = [0.25, 0.75]
    if model is not None:
        node["description_embedding_model"] = model
    path.write_text(json.dumps({"nodes": [node], "edges": []}), encoding="utf-8")
    if storage == "sidecar":
        path.with_suffix(".emb.json").write_text(
            json.dumps(
                {
                    "home": [0.25, 0.75]
                    if model is None
                    else {"model": model, "embedding": [0.25, 0.75]}
                }
            ),
            encoding="utf-8",
        )
    manager = GraphManager(
        page_detail_client=router_client,
        embedding_client=router_client,
        page_detail_model="openai/gpt-6.1-sol",
        embedding_model="google/gemini-embedding-2",
    )
    manager.load_graph(path)
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=embedding_body),
        ]
        if operation == "identify"
        else [httpx.Response(200, json=embedding_body)]
    )

    def compare() -> None:
        if operation == "find":
            manager.find_node_by_description("home")
        else:
            manager.identify_state("com.demo.Home", png_screenshot)

    for _ in range(2):
        with pytest.raises(ValueError, match=r"Text embedding.*Rebuild or re-explore"):
            compare()
        assert router_requests == []


def test_text_embedding_model_survives_save_load(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    manager = GraphManager(
        embedding_client=router_client, embedding_model="google/gemini-embedding-2"
    )
    manager.graph.add_node(
        "home",
        description_embedding=[0.25, 0.75],
        description_embedding_model=manager.embedding_model,
    )
    path = tmp_path / "demo.json"
    manager.save_graph(path)
    saved = json.loads(path.with_suffix(".emb.json").read_text(encoding="utf-8"))
    assert saved["home"]["model"] == "google/gemini-embedding-2"
    manager.load_graph(path)
    router_responses.append(httpx.Response(200, json=embedding_body))
    assert manager.find_node_by_description("home") == [("home", pytest.approx(1.0))]


@pytest.mark.parametrize("recover", [False, True])
def test_sdk_validation_uses_bounded_application_recovery(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    from openrouter.errors import ResponseValidationError

    bad = {key: value for key, value in embedding_body.items() if key != "model"}
    router_responses.extend(httpx.Response(200, json=bad) for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(httpx.Response(200, json=embedding_body))

    def request() -> object:
        return embedding_cache.compute_embedding_with_retry(
            "test",
            png_screenshot,
            model="model",
            base_url="https://openrouter.ai/api/v1",
            app_name="demo",
            node_id="n1",
            client=router_client,
        )

    if recover:
        assert request() == [0.25, 0.75]
    else:
        with pytest.raises(ResponseValidationError) as caught:
            request()
        assert caught.value.status_code == 200
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("role", ["embedding", "image_embedding"])
@pytest.mark.parametrize("entry", ["resolver", "factory", "offline", "runtime"])
@pytest.mark.usefixtures("model_environment")
def test_legacy_native_url_rejected_before_requests(
    role: str,
    entry: str,
    tmp_path: Path,
    router_requests: list[httpx.Request],
    router_http: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph.commands import embed

    secret = "dummy-google-secret-never-log"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")
    cfg = {
        "native_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "${GEMINI_API_KEY}",
        "model": "gemini-embedding-2",
    }

    def configure() -> None:
        if entry == "factory":
            make_client(cfg, role=role, http_client=router_http)
        elif entry == "resolver":
            resolve_image_embedding_settings({role: cfg})
        elif entry == "runtime":
            aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config={role: cfg})
        else:
            path = tmp_path / "legacy.yaml"
            path.write_text(
                f"vlm:\n  {role}:\n    native_base_url: https://generativelanguage.googleapis.com/v1beta\n    api_key: '${{GEMINI_API_KEY}}'\n    model: gemini-embedding-2\n"
            )
            embed.load_image_embedding_settings(
                path,
                graphs_override=None,
                api_key_override=None,
                model_override=None,
                base_url_override=None,
            )

    with pytest.raises(ValueError, match=rf"{role}\.native_base_url") as caught:
        configure()
    assert secret not in str(caught.value)
    assert "base_url" in str(caught.value)
    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert router_requests == []


@pytest.mark.parametrize("exceptional", [False, True])
def test_owned_http_client_closed_on_sdk_scope_exit(exceptional: bool) -> None:
    client, _ = make_client({"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action")
    owned = client.sdk_configuration.client
    assert isinstance(owned, httpx.Client)
    try:
        with client:
            if exceptional:
                raise RuntimeError("scope failed")
    except RuntimeError:
        assert exceptional
    assert owned.is_closed


def test_borrowed_client_and_hooks_survive_sdk_scope(
    router_http: httpx.Client, router_responses: list[httpx.Response], chat_body: dict[str, object]
) -> None:
    events: list[str] = []
    router_http.event_hooks["response"].append(lambda _response: events.append("response"))
    client, model = make_client(
        {"model": "openai/gpt-6.1-sol", "api_key": "test"},
        role="page_detail",
        http_client=router_http,
    )
    router_responses.append(httpx.Response(200, json=chat_body))
    with client:
        vlm_utils.send_chat(client, model=model, messages=[])
    assert not router_http.is_closed
    assert events == ["response"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_screenshot_batch_reuses_and_closes_client_with_partial_progress(
    interrupted: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    from collections.abc import Iterator

    graph = tmp_path / "demo.json"
    cache: dict[str, list[float]] = {}
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
        ]
    )

    def candidates() -> Iterator[tuple[str, str]]:
        yield "n1", "shot-1"
        if interrupted:
            raise KeyboardInterrupt
        yield "n2", "shot-2"

    def compute() -> None:
        embedding_cache.compute_missing_image_embeddings(
            graph,
            cache,
            candidates(),
            api_key="test",
            model="model",
            base_url="http://localhost:8080/v1",
            app_name="demo",
        )

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            compute()
    else:
        compute()
    assert len(router_requests) == (2 if interrupted else 3)
    assert no_sleep == [2.0]
    assert len(owned_router_clients) == 1
    assert all(client.is_closed for client in owned_router_clients)
    assert (
        embedding_cache.load_image_embeddings(graph, model="model", node_ids={"n1", "n2"}) == cache
    )
    assert set(cache) == ({"n1"} if interrupted else {"n1", "n2"})


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_v2_invoked_callback_closes_clients_and_reuses_image_client(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    assert all(client.is_closed for client in owned_router_clients)
    initial_count = len(owned_router_clients)
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(4))

    def step(_task: str, _state: dict[str, object], _history: dict[str, object]) -> str:
        for screenshot in ("one", "two"):
            assert built._compute_runtime_image_embedding_with_retry(
                screenshot, "demo", "current"
            ) == [0.25, 0.75]
        if exceptional:
            raise RuntimeError("callback failed")
        return "ok"

    monkeypatch.setattr(built, "_step", step)
    for _ in range(2):
        if exceptional:
            with pytest.raises(RuntimeError, match="callback failed"):
                built.to_agent("task", {}, {})
        else:
            assert built.to_agent("task", {}, {}) == "ok"
        assert all(client.is_closed for client in owned_router_clients)
    assert len(owned_router_clients) == initial_count + 8


@pytest.mark.usefixtures("model_environment")
def test_v2_partial_initialization_closes_created_clients(
    tmp_path: Path, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match="page_detail"):
        aitk_translator.UIKobeV2Translator(
            graph_dir=str(tmp_path), vlm_config={"action": {"api_key": "test"}}
        )
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_audit_cli_closes_owned_clients_on_exit(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph.commands import audit

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    app_dir.mkdir(parents=True)
    (app_dir / "demo.json").write_text('{"nodes":[],"edges":[]}')
    config = tmp_path / "config.yaml"
    config.write_text(
        f"experiment:\n  graph_dir: {graph_dir}\napps:\n  - name: demo\n    package_name: com.demo\n"
    )

    def audit_graph(_graph: GraphManager, _app: str, _path: Path) -> dict[str, object]:
        if exceptional:
            raise RuntimeError("audit failed")
        return {"issues": []}

    monkeypatch.setattr(audit, "run_audit", audit_graph)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)
    if exceptional:
        with pytest.raises(RuntimeError, match="audit failed"):
            audit.main(["-c", str(config)])
    else:
        assert audit.main(["-c", str(config)]) == 0
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("borrowed", [False, True])
def test_factory_initialization_failure_keeps_correct_owner(
    borrowed: bool,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph import utils

    supplied = httpx.Client() if borrowed else None

    def fail_sdk(**_kwargs: object) -> OpenRouter:
        raise RuntimeError("SDK construction failed")

    monkeypatch.setattr(utils, "ConfiguredOpenRouter", fail_sdk)
    with pytest.raises(RuntimeError, match="SDK construction failed"):
        make_client(
            {"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action", http_client=supplied
        )
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed is not borrowed
    if supplied is not None:
        supplied.close()


@pytest.mark.parametrize(
    "role", ["page_detail", "instruction", "action", "embedding", "image_embedding"]
)
def test_missing_key_names_explicit_configuration_role(role: str) -> None:
    with pytest.raises(ValueError, match=rf"{role}\.api_key.*OPENROUTER_API_KEY"):
        make_client({"model": "openai/gpt-6.1-sol"}, role=role)


def test_image_batches_keep_endpoint_credentials_and_models_separate(
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
) -> None:
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(4))
    for index in range(2):
        result = embedding_cache.compute_missing_image_embeddings(
            tmp_path / f"graph-{index}.json",
            {},
            [("n1", "one"), ("n2", "two")],
            api_key=f"key-{index}",
            model=f"model-{index}",
            base_url=f"https://endpoint-{index}.example/v1",
            app_name="demo",
        )
        assert result.computed == 2
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)
    for index, request in enumerate(router_requests):
        setting = index // 2
        assert request.headers["authorization"] == f"Bearer key-{setting}"
        assert str(request.url) == f"https://endpoint-{setting}.example/v1/embeddings"
        assert json.loads(request.content)["model"] == f"model-{setting}"

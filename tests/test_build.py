"""Tests for app-graph-build: a graph built from a recorded step log, with every model call
answered by a fake and no device."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.commands import build, plot
from android_app_graph.graph_files import identify_file_path, read_json_lines
from android_app_graph.payloads import as_object_dict
from android_app_graph.step_log import (
    Observation,
    StepRecorder,
    Transition,
    read_records,
    save_image_embeddings,
)
from android_app_graph.utils import MatchingSettings, ModelSettings, graph_manager
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE, GraphManager
from android_app_graph.utils.vlm_utils import DescribedScreenshot, DescribeRequest

APP = "com.demo"
OTHER_APP = "com.other"
IMAGE_MODEL = "google/gemini-embedding-2"
TAP = {"action": "tap", "x": 10, "y": 20}
SCROLL = {"action": "swipe", "direction": "up"}
INSTRUCTION = "Open the next screen"


def _png(shade: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True)
class _Page:
    """A screen of the recorded app: the page it shows, which the fake verifier compares,
    its description and state, and the image the step log keeps."""

    page: int
    shade: int
    state: dict[str, object] = field(default_factory=dict)
    package: str = APP
    external_web: bool = False

    @property
    def description(self) -> str:
        return f"Screen {self.page}"

    @property
    def image_vector(self) -> list[float]:
        """Pages ten degrees apart, so the image source ranks the nearest pages first."""
        angle = math.radians(10 * self.page)
        return [math.cos(angle), math.sin(angle)]


SHARE_MENU: dict[str, object] = {"open_menu": "share menu"}
PAGES = [
    _Page(page, shade=20 + 20 * page, state=SHARE_MENU if page == 7 else {}) for page in range(9)
]
SCROLLED = _Page(6, shade=230, state={"first_visible_item": "row 5"})
OTHER = _Page(99, shade=250, package=OTHER_APP)
WEB = _Page(98, shade=10, external_web=True)


class _Device:
    w = 100
    h = 200

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self.page = PAGES[0]

    def get_state(self, **_kwargs: float | None) -> dict[str, Any]:
        page = self.page
        return {
            "activity": f"{page.package}.Main",
            "package": page.package,
            "screenshot": _png(page.shade),
        }

    def exe_action(self, action: dict[str, Any]) -> None:
        del action  # the next screen is set by the recording


@dataclass
class _Recording:
    """Records a run in a step log as exploration does, with the image vector of each
    screenshot it matches, by the page it shows."""

    recorder: StepRecorder
    device: _Device
    vectors: dict[str, list[float]]
    pages: dict[str, _Page]

    def show(self, page: _Page) -> Observation:
        self.device.page = page
        observation = Observation.from_state(self.recorder.get_state())
        self.pages[observation.observation_id] = page
        return observation

    def match(self, observation: Observation, expected: Observation | None = None) -> None:
        self.recorder.record_match(observation, expected.observation_id if expected else None)
        self.vectors[observation.observation_id] = self.pages[
            observation.observation_id
        ].image_vector

    def step(
        self, before: Observation, page: _Page, *, action: dict[str, Any] = TAP, match: bool = True
    ) -> Observation:
        self.recorder.exe_action(action)
        after = self.show(page)
        if match:
            self.match(after)
        self.recorder.record_transition(
            Transition(
                before=before,
                after=after,
                instruction=INSTRUCTION,
                element=None,
                action=action,
                from_node="the node of the explored graph",
                to_node="the node of the explored graph",
            )
        )
        return after


@dataclass
class _Log:
    steps: Path
    pages: dict[str, _Page]
    observations: dict[str, Observation]


def _record_log(steps: Path) -> _Log:
    """Record an exploration run through seven new screens, a scroll and a tap after it, a
    step into another app from the share menu open on the screen it reaches and back, a
    step planned on a screenshot read after the match, a
    replay that reaches the start screen, a step from it and one onto an outside web page
    and back; then an audit run with one step."""
    device = _Device()
    vectors: dict[str, list[float]] = {}
    pages: dict[str, _Page] = {}
    seen: dict[str, Observation] = {}
    run = _Recording(
        StepRecorder(device, steps, run_kind="explore", source="main_loop"), device, vectors, pages
    )
    start = seen["s0"] = run.show(PAGES[0])
    run.match(start)
    screen = start
    for page in PAGES[1:7]:
        screen = seen[f"s{page.page}"] = run.step(screen, page)
    scrolled = seen["scrolled"] = run.step(screen, SCROLLED, action=SCROLL)
    screen = seen["s7"] = run.step(scrolled, PAGES[7])
    run.step(screen, OTHER, match=False)
    run.recorder.exe_action({"action": "back"})
    back = seen["back"] = run.show(PAGES[7])
    run.match(back, expected=screen)
    planned = run.show(PAGES[7])
    seen["after_planned"] = run.step(planned, PAGES[1])
    with run.recorder.source("navigation_replay"):
        replayed = seen["replayed"] = run.show(PAGES[0])
        run.match(replayed, expected=start)
    planned = run.show(PAGES[0])
    screen = seen["after_replay"] = run.step(planned, PAGES[2])
    seen["web"] = run.step(screen, WEB)
    with run.recorder.source("external_web_recovery"):
        run.recorder.exe_action({"action": "back"})
        seen["off_web"] = run.show(PAGES[2])
        run.match(seen["off_web"], expected=screen)
    audit = _Recording(
        StepRecorder(device, steps, run_kind="audit", source="re_explore"), device, vectors, pages
    )
    audit_start = audit.show(PAGES[3])
    audit.match(audit_start)
    seen["audit"] = audit.step(audit_start, PAGES[8])
    save_image_embeddings(steps, vectors, model=IMAGE_MODEL)
    return _Log(steps, pages, seen)


@dataclass
class _FakeVlm:
    """Answers every model call a build makes from the pages the screenshots show."""

    by_image: dict[str, _Page]
    verifier_calls: int = 0
    merge_audits: int = 0
    calls: list[str] = field(default_factory=list)

    def describe(
        self,
        _client: OpenRouter | None,
        request: DescribeRequest,
        *,
        settings: ModelSettings | None = None,
    ) -> tuple[str, dict[str, object], list[dict[str, str]], bool]:
        del settings
        page = self.by_image[request.screenshot_b64]
        return page.description, dict(page.state), [], page.external_web

    def verify(
        self,
        _client: OpenRouter | None,
        *,
        existing: DescribedScreenshot,
        new: DescribedScreenshot,
        settings: ModelSettings | None = None,
    ) -> dict[str, Any]:
        del settings
        self.verifier_calls += 1
        same = self.by_image[new.screenshot_b64].page == self.by_image[existing.screenshot_b64].page
        return {"same": same, "reason": "the same page" if same else "another page"}

    def embed(
        self, _client: OpenRouter | None, text: str, model: str = "", request_timeout: float = 0.0
    ) -> list[float]:
        """Pages of one parity are described alike, so the text source proposes other
        nodes than the image source."""
        del model, request_timeout
        return [1.0, 0.0] if int(text.rsplit(maxsplit=1)[-1]) % 2 == 0 else [0.0, 1.0]

    def audit_merge_nodes(
        self,
        _client: OpenRouter | None,
        graph_summary: str,
        app_name: str = "",
        settings: ModelSettings | None = None,
    ) -> dict[str, Any]:
        del graph_summary, app_name, settings
        self.merge_audits += 1
        return {"issues": [], "summary": "no duplicates"}


def _webp_b64(steps: Path, observation_id: str) -> str:
    return base64.b64encode((steps / f"{observation_id}.webp").read_bytes()).decode("ascii")


@pytest.fixture
def recorded(tmp_path: Path) -> _Log:
    return _record_log(tmp_path / "explored" / "demo" / "steps")


@pytest.fixture
def vlm(recorded: _Log, monkeypatch: pytest.MonkeyPatch) -> _FakeVlm:
    fake = _FakeVlm({_webp_b64(recorded.steps, obs): page for obs, page in recorded.pages.items()})
    monkeypatch.setattr(graph_manager, "describe_page_and_state", fake.describe)
    monkeypatch.setattr(graph_manager, "verify_same_node", fake.verify)
    monkeypatch.setattr(graph_manager, "get_embedding", fake.embed)
    monkeypatch.setattr(graph_manager, "audit_merge_nodes", fake.audit_merge_nodes)

    def no_image_embedding(*_args: object, **_kwargs: object) -> list[float]:
        msg = "a build embeds no screenshot"
        raise AssertionError(msg)

    monkeypatch.setattr(graph_manager, "get_image_embedding", no_image_embedding)
    return fake


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # setup_logging replaces the root handlers, caplog's among them.
    monkeypatch.setattr(build, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("APP_GRAPH_PAGE_DETAIL_MODEL", "openai/gpt-6.1-sol")
    monkeypatch.setenv("APP_GRAPH_EMBEDDING_MODEL", IMAGE_MODEL)
    return _write_config(tmp_path / "build.yaml", verifier_calls=3)


def _write_config(path: Path, *, verifier_calls: int, checkpoint_steps: int = 50) -> Path:
    path.write_text(
        json.dumps(
            {
                "experiment": {"coverage_checkpoint_steps": checkpoint_steps},
                "apps": [{"name": "demo", "package_name": APP}],
                "vlm": {"matching": {"verifier_calls": verifier_calls}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _build(config: Path, recorded: _Log, out: Path, *extra: str) -> int:
    return build.main(
        [
            "-c",
            str(config),
            "--app",
            "demo",
            "--steps",
            str(recorded.steps),
            "--out",
            str(out),
            *extra,
        ]
    )


def _load(out: Path, matching: MatchingSettings) -> GraphManager:
    graph = GraphManager(
        embedding=ModelSettings(IMAGE_MODEL), image_embedding=ModelSettings(IMAGE_MODEL)
    )
    graph.matching = matching
    graph.load_graph(out / "demo.json")
    return graph


@pytest.mark.usefixtures("config")
def test_builds_at_five_verifier_calls_make_more_calls_and_leave_the_step_log_as_it_was(
    recorded: _Log, vlm: _FakeVlm, tmp_path: Path
) -> None:
    before = _hashes(recorded.steps)
    calls: dict[int, int] = {}
    for limit in (3, 5):
        vlm.verifier_calls = 0
        limited = _write_config(tmp_path / f"build{limit}.yaml", verifier_calls=limit)

        assert _build(limited, recorded, tmp_path / f"built{limit}") == 0

        calls[limit] = vlm.verifier_calls
        built = _load(tmp_path / f"built{limit}", MatchingSettings(verifier_calls=limit))
        assert built.matching.verifier_calls == limit

    assert calls[5] > calls[3]
    assert _hashes(recorded.steps) == before


def test_a_build_writes_its_graph_into_a_new_directory_linked_to_the_step_log(
    recorded: _Log, vlm: _FakeVlm, config: Path, tmp_path: Path
) -> None:
    out = tmp_path / "built"

    assert _build(config, recorded, out) == 0

    assert sorted(path.name for path in out.iterdir()) == [
        "demo.json",
        "demo_identify.jsonl",
        "demo_text_emb.json",
        "steps",
    ]
    assert (out / "steps").is_symlink()
    assert (out / "steps").resolve() == recorded.steps.resolve()
    lines = read_json_lines(identify_file_path(out / "demo.json"))
    matches = [
        record["observation"]
        for record in read_records(recorded.steps)
        if (record["kind"], record["run_kind"]) == ("match", "explore")
    ]
    assert [line["observation"] for line in lines] == matches
    assert vlm.merge_audits == 0
    _data, plotted = plot.load_graph(str(out / "demo.json"))
    assert plotted.number_of_nodes() == len(_load(out, MatchingSettings()).graph)


@pytest.mark.usefixtures("vlm")
def test_a_build_places_screens_and_steps_as_exploration_did(
    recorded: _Log, config: Path, tmp_path: Path
) -> None:
    out = tmp_path / "built"
    assert _build(config, recorded, out) == 0
    graph = _load(out, MatchingSettings())
    seen = recorded.observations

    def node(name: str) -> str:
        found = graph.node_of(seen[name].observation_id)
        assert found is not None
        return found

    assert len({node(f"s{page}") for page in range(8)}) == 8
    assert node("scrolled") == node("s6")
    assert node("back") == node("s7")
    loop = graph.graph[node("s6")][node("s6")]
    assert loop["schema_deltas"] == [{"first_visible_item": {"before": None, "after": "row 5"}}]
    assert graph.graph[node("s6")][node("s7")]["setups"] == [[SCROLL]]
    assert graph.graph[node("s6")][node("s7")]["open_menus"] == [None]
    ext = graph.graph[node("s7")][f"ext_{OTHER_APP}"]
    assert ext["target_observations"] == [f"[left app to {OTHER_APP}]"]
    assert ext["open_menus"] == ["share menu"]
    assert graph.graph.has_edge(node("s7"), node("after_planned"))
    assert graph.graph[node("s0")][node("after_replay")]["setups"] == [[]]
    web = graph.graph[node("after_replay")][EXTERNAL_WEB_NODE]
    assert web["target_observations"] == [WEB.description]
    assert node("off_web") == node("after_replay")
    assert graph.node_of(seen["audit"].observation_id) is None
    (back,) = [
        line
        for line in read_json_lines(identify_file_path(out / "demo.json"))
        if line["observation"] == seen["back"].observation_id
    ]
    decision = as_object_dict(back["identify"])
    assert (decision["stage"], decision["expected_node"]) == ("exact", node("s7"))


@pytest.mark.usefixtures("vlm")
def test_a_build_replays_the_audit_runs_when_asked(
    recorded: _Log, config: Path, tmp_path: Path
) -> None:
    out = tmp_path / "built"

    assert _build(config, recorded, out, "--audit-runs") == 0

    graph = _load(out, MatchingSettings())
    audited = recorded.observations["audit"].observation_id
    assert graph.node_of(audited) is not None
    assert graph.description(audited) == "Screen 8"


@pytest.mark.usefixtures("config")
def test_a_build_runs_the_merge_audit_at_the_coverage_checkpoint_interval(
    recorded: _Log, vlm: _FakeVlm, tmp_path: Path
) -> None:
    every_four = _write_config(tmp_path / "every4.yaml", verifier_calls=3, checkpoint_steps=4)

    assert _build(every_four, recorded, tmp_path / "built") == 0

    # Twelve exploration steps: checkpoints after the fourth and the eighth.
    assert vlm.merge_audits == 2


@pytest.mark.usefixtures("vlm")
def test_a_build_stops_naming_how_many_matched_screenshots_have_no_image_vector(
    recorded: _Log, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vectors = json.loads((recorded.steps / "image_emb.json").read_text(encoding="utf-8"))
    for observation in ("s1", "s2"):
        del vectors["embeddings"][recorded.observations[observation].observation_id]
    (recorded.steps / "image_emb.json").write_text(json.dumps(vectors), encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        _build(config, recorded, tmp_path / "built")

    assert caught.value.code == 2
    assert "2 of the 15 screenshot(s) the log matches have no image vector" in (
        capsys.readouterr().err
    )
    assert not (tmp_path / "built").exists()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        pytest.param("out exists", "exists; a build writes into a new directory", id="out exists"),
        pytest.param("no steps", "No step log at", id="no step log"),
        pytest.param("no app", "no app named 'demo'", id="app not configured"),
        pytest.param(
            "unrecorded", "match records of observations without a record", id="unrecorded"
        ),
        pytest.param("no config", "config file not found", id="no config"),
    ],
)
@pytest.mark.usefixtures("vlm")
def test_a_build_rejects_inputs_it_cannot_use_before_writing_anything(
    problem: str,
    message: str,
    recorded: _Log,
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "built"
    steps = recorded.steps
    if problem == "out exists":
        out.mkdir()
    elif problem == "no steps":
        steps = tmp_path / "missing" / "steps"
    elif problem == "no app":
        config.write_text(json.dumps({"apps": [{"name": "other", "package_name": "x"}]}), "utf-8")
    elif problem == "unrecorded":
        (steps / f"{recorded.observations['s3'].observation_id}.json").unlink()
    else:
        config = tmp_path / "missing.yaml"
    before = _hashes(recorded.steps)

    with pytest.raises(SystemExit) as caught:
        build.main(["-c", str(config), "--app", "demo", "--steps", str(steps), "--out", str(out)])

    assert caught.value.code == 2
    assert message in capsys.readouterr().err
    assert _hashes(recorded.steps) == before
    assert out.exists() == (problem == "out exists")


def test_a_build_skips_a_step_with_no_matched_start_or_an_unmatched_screen_of_the_app(
    vlm: _FakeVlm, config: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps = tmp_path / "log" / "steps"
    device = _Device()
    run = _Recording(
        StepRecorder(device, steps, run_kind="explore", source="main_loop"), device, {}, {}
    )
    first = run.show(PAGES[0])
    second = run.step(first, PAGES[1], match=False)
    run.match(second)
    run.step(second, PAGES[2], match=False)
    save_image_embeddings(steps, run.vectors, model=IMAGE_MODEL)
    vlm.by_image = {_webp_b64(steps, obs): page for obs, page in run.pages.items()}
    log = _Log(steps, run.pages, {})

    assert _build(config, log, tmp_path / "built") == 0

    assert "starts on no screenshot the build matched" in caplog.text
    assert "ends on a screen of the app that was not matched" in caplog.text
    graph = _load(tmp_path / "built", MatchingSettings())
    assert graph.graph.number_of_edges() == 0

"""app-graph-plot argument handling and its pure helpers, exercised without a browser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from android_app_graph.commands import plot


def test_plot_parser_defaults() -> None:
    args = plot.build_parser().parse_args(["graph.json"])
    assert args.graph == "graph.json"
    assert args.output is None
    assert args.backend == "pyvis"
    assert args.no_open is False
    assert args.trim_leaves == 1


def test_plot_rejects_unknown_backend(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        plot.build_parser().parse_args(["graph.json", "--backend", "bogus"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_plot_reports_missing_viz_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The submodules too, not just the packages: import_module returns a cached
    # submodule without consulting its parent, and under random ordering a
    # pyvis-importing test may already have populated that cache.
    for name in ("pyvis", "pyvis.network", "matplotlib", "matplotlib.pyplot"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(SystemExit) as excinfo:
        plot.main(["graph.json", "--no-open"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "android-app-graph[viz]" in err
    assert "missing module: pyvis.network" in err


def test_plot_reports_missing_graph_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing graph file exits 2 through parser.error, not a print and return 0."""
    with pytest.raises(SystemExit) as excinfo:
        plot.main([str(tmp_path / "absent.json"), "--no-open"])
    assert excinfo.value.code == 2
    assert "graph file not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        (None, ""),
        ("  one   two  ", "one two"),
        ("one two three four five six seven", "one two three four five six..."),
    ],
)
def test_shorten_words(text: str | None, expected: str) -> None:
    assert plot._shorten_words(text) == expected


def test_shorten_words_truncates_on_characters() -> None:
    assert plot._shorten_words("abcdefghij", max_words=6, max_chars=8) == "abcde..."


def test_shorten_edge_label_strips_noise() -> None:
    label = plot._shorten_edge_label("Tap the [Search] button to open {{value}} (again)")
    assert "[" not in label
    assert "(" not in label
    assert "button" not in label
    assert "Tap" in label


def test_wrap_label_breaks_on_width() -> None:
    assert plot._wrap_label("alpha beta gamma", width=10) == "alpha beta\\ngamma"


def test_wrap_label_of_empty_text() -> None:
    assert plot._wrap_label("") == ""


@pytest.mark.parametrize(
    ("description", "category"),
    [
        ("Search results list", "search"),
        ("Settings menu", "settings"),
        ("Article detail", "detail"),
        ("Home map", "home"),
        ("Something else entirely", "other"),
    ],
)
def test_node_category(description: str, category: str) -> None:
    assert plot._node_category(description) == category


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        ([], "?"),
        ([{"action": "tap", "x": 1, "y": 2}], "tap (1, 2)"),
        ([{"action": "type", "text": "hi"}], "type 'hi'"),
        (
            [{"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4}],
            "swipe (1,2)→(3,4)",
        ),
        ([{"action": "back"}], "back"),
        ([[{"action": "tap", "x": 1, "y": 2}, {"action": "back"}]], "tap (1, 2) → back"),
    ],
)
def test_summarize_actions(actions: list[Any], expected: str) -> None:
    assert plot._summarize_actions(actions) == expected


def _paper_graph() -> nx.DiGraph:
    """Two connected nodes, one external node and one disconnected leaf."""
    G = nx.DiGraph()
    G.add_node("s0_home", page_description="Home")
    G.add_node("s1_search", page_description="Search results")
    G.add_node("ext_launcher", page_description="Launcher")
    G.add_node("s2_orphan", page_description="Orphan")
    G.add_edge("s0_home", "s1_search", label="open search", visit_count=2)
    G.add_edge("s1_search", "s0_home", label="go back", visit_count=1)
    G.add_edge("s0_home", "ext_launcher", label="leave app")
    return G


def test_largest_weak_component_of_empty_graph() -> None:
    assert plot._largest_weak_component(nx.DiGraph()).number_of_nodes() == 0


def test_largest_weak_component_keeps_the_biggest() -> None:
    kept = plot._largest_weak_component(_paper_graph())
    assert set(kept) == {"s0_home", "s1_search", "ext_launcher"}


def test_drop_external_nodes() -> None:
    kept = plot._drop_external_nodes(_paper_graph())
    assert "ext_launcher" not in kept
    assert "s0_home" in kept


def test_trim_leaf_rounds_removes_leaves() -> None:
    trimmed = plot._trim_leaf_rounds(_paper_graph(), 1)
    assert "s2_orphan" not in trimmed
    assert "ext_launcher" not in trimmed


def test_trim_leaf_rounds_is_a_no_op_without_rounds() -> None:
    graph = _paper_graph()
    assert set(plot._trim_leaf_rounds(graph, 0)) == set(graph)


def test_prepare_paper_graph_filters_and_reports(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=plot.__name__):
        prepared = plot._prepare_paper_graph(
            _paper_graph(),
            keep_disconnected=False,
            keep_external=False,
            trim_leaves=1,
        )
    assert set(prepared) == {"s0_home", "s1_search"}
    assert "Paper view: kept" in caplog.text


def test_prepare_paper_graph_can_keep_everything() -> None:
    graph = _paper_graph()
    prepared = plot._prepare_paper_graph(
        graph,
        keep_disconnected=True,
        keep_external=True,
        trim_leaves=0,
    )
    assert set(prepared) == set(graph)


def test_as_digraph_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="expected a networkx DiGraph"):
        plot._as_digraph(object())


def _write_graph(tmp_path: Path, *, stem: str = "demo") -> Path:
    graph_path = tmp_path / f"{stem}.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "s0_home",
                        "page_description": "Home",
                        "activity": "com.demo.Home",
                        "visit_count": 3,
                        "state_schema": {"tab": ["map"]},
                        "interactable_elements": [{"explored": True}, {"explored": False}],
                    },
                    {
                        "id": "s1_search",
                        "page_description": "Search results",
                        "activity": "com.demo.Search",
                    },
                ],
                "edges": [
                    {
                        "source": "s0_home",
                        "target": "s1_search",
                        "actions": [{"action": "tap", "x": 5, "y": 6}],
                        "instructions": ["open search"],
                        "num_steps": [1],
                        "visit_count": 2,
                    },
                    {
                        "source": "s1_search",
                        "target": "s1_search",
                        "actions": [{"action": "type", "text": "cafe"}],
                        "schema_deltas": [{"query": {"before": None, "after": "cafe"}}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def test_load_graph_reads_nodes_edges_and_screenshots(tmp_path: Path) -> None:
    graph_path = _write_graph(tmp_path)
    screenshots = tmp_path / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "s0_home.png").write_bytes(b"not-really-a-png")

    data, graph = load_result = plot.load_graph(str(graph_path))
    assert len(load_result) == 2
    assert data["nodes"][0]["id"] == "s0_home"
    assert set(graph) == {"s0_home", "s1_search"}
    assert graph.nodes["s0_home"]["elements_explored"] == 1
    assert graph.nodes["s0_home"]["elements_total"] == 2
    assert graph.nodes["s0_home"]["screenshot_uri"].startswith("data:image/png;base64,")
    assert graph.nodes["s1_search"]["screenshot_uri"] is None
    assert graph.edges["s0_home", "s1_search"]["label"].startswith("open search")
    # The self-loop edge falls back to an action summary plus its schema delta.
    self_loop_label = graph.edges["s1_search", "s1_search"]["label"]
    assert self_loop_label.startswith("type 'cafe'")
    assert "query" in self_loop_label


def test_load_graph_falls_back_to_the_unaudited_screenshots(tmp_path: Path) -> None:
    graph_path = _write_graph(tmp_path, stem="demo_audited")
    screenshots = tmp_path / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "s0_home.png").write_bytes(b"not-really-a-png")

    _data, graph = plot.load_graph(str(graph_path))
    assert graph.nodes["s0_home"]["screenshot_uri"] is not None


def test_plot_pyvis_writes_fullscreen_html(tmp_path: Path) -> None:
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "graph.html"
    plot.plot_pyvis(graph, str(output))
    html = output.read_text(encoding="utf-8")
    assert "#mynetwork" in html
    assert "100vh" in html


def test_plot_matplotlib_writes_an_image(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("agg")
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "graph.png"
    plot.plot_matplotlib(graph, str(output))
    assert output.stat().st_size > 0


def test_plot_paper_matplotlib_writes_an_image(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("agg")
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "paper.png"
    plot.plot_paper_matplotlib(graph, str(output), layout="spring")
    assert output.stat().st_size > 0


def test_plot_paper_graphviz_falls_back_without_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("agg")
    monkeypatch.setattr(plot.shutil, "which", lambda _cmd: None)
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "paper.png"
    plot.plot_paper_graphviz(graph, str(output))
    assert output.stat().st_size > 0
    assert not output.with_suffix(".dot").exists()


def test_plot_paper_graphviz_invokes_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "paper.svg"
    recorded: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> None:
        recorded.append(argv)
        output.write_bytes(b"<svg/>")

    monkeypatch.setattr(plot.shutil, "which", lambda _cmd: "/usr/bin/dot")
    monkeypatch.setattr(plot.subprocess, "run", fake_run)

    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    plot.plot_paper_graphviz(graph, str(output))

    dot_path = output.with_suffix(".dot")
    assert dot_path.exists()
    assert "digraph G {" in dot_path.read_text(encoding="utf-8")
    assert recorded[0][0] == "/usr/bin/dot"
    assert recorded[0][1] == "-Tsvg"

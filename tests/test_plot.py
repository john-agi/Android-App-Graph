"""app-graph-plot argument handling and its pure helpers, exercised without a browser."""

from __future__ import annotations

import base64
import dataclasses
import json
import re
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
    assert plot.shorten_words(text) == expected


def test_shorten_words_truncates_on_characters() -> None:
    assert plot.shorten_words("abcdefghij", max_words=6, max_chars=8) == "abcde..."


def test_shorten_edge_label_strips_noise() -> None:
    label = plot.shorten_edge_label("Tap the [Search] button to open {{value}} (again)")
    assert "[" not in label
    assert "(" not in label
    assert "button" not in label
    assert "Tap" in label


def test_wrap_label_breaks_on_width() -> None:
    assert plot.wrap_label("alpha beta gamma", width=10) == "alpha beta\\ngamma"


def test_wrap_label_of_empty_text() -> None:
    assert plot.wrap_label("") == ""


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
    assert plot.node_category(description) == category


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
    assert plot.summarize_actions(actions) == expected


def _paper_graph() -> nx.DiGraph:
    """Two connected nodes, one external node and one disconnected leaf."""
    graph = nx.DiGraph()
    graph.add_node("s0_home", page_description="Home")
    graph.add_node("s1_search", page_description="Search results")
    graph.add_node("ext_launcher", page_description="Launcher")
    graph.add_node("s2_orphan", page_description="Orphan")
    graph.add_edge("s0_home", "s1_search", label="open search", visit_count=2)
    graph.add_edge("s1_search", "s0_home", label="go back", visit_count=1)
    graph.add_edge("s0_home", "ext_launcher", label="leave app")
    return graph


def test_largest_weak_component_of_empty_graph() -> None:
    assert plot.largest_weak_component(nx.DiGraph()).number_of_nodes() == 0


def test_largest_weak_component_keeps_the_biggest() -> None:
    kept = plot.largest_weak_component(_paper_graph())
    assert set(kept) == {"s0_home", "s1_search", "ext_launcher"}


def test_drop_external_nodes() -> None:
    kept = plot.drop_external_nodes(_paper_graph())
    assert "ext_launcher" not in kept
    assert "s0_home" in kept


def test_trim_leaf_rounds_removes_leaves() -> None:
    trimmed = plot.trim_leaf_rounds(_paper_graph(), 1)
    assert "s2_orphan" not in trimmed
    assert "ext_launcher" not in trimmed


def test_trim_leaf_rounds_is_a_no_op_without_rounds() -> None:
    graph = _paper_graph()
    assert set(plot.trim_leaf_rounds(graph, 0)) == set(graph)


def test_prepare_paper_graph_filters_and_reports(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=plot.__name__):
        prepared = plot.prepare_paper_graph(
            _paper_graph(),
            keep_disconnected=False,
            keep_external=False,
            trim_leaves=1,
        )
    assert set(prepared) == {"s0_home", "s1_search"}
    assert "Paper view: kept" in caplog.text


def test_prepare_paper_graph_can_keep_everything() -> None:
    graph = _paper_graph()
    prepared = plot.prepare_paper_graph(
        graph,
        keep_disconnected=True,
        keep_external=True,
        trim_leaves=0,
    )
    assert set(prepared) == set(graph)


def test_as_digraph_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="expected a networkx DiGraph"):
        plot.as_digraph(object())


def _saved(edge: dict[str, Any]) -> dict[str, Any]:
    """Return ``edge`` with the per-action lists ``add_edge`` records for its actions, for
    each list it does not hold."""
    count = len(edge["actions"])
    recorded = {
        "num_steps": [1] * count,
        "instructions": [""] * count,
        "target_observations": [""] * count,
        "schema_deltas": [None] * count,
        "setups": [None] * count,
    }
    return {**recorded, **edge}


def _write_graph(tmp_path: Path, *, stem: str = "demo") -> Path:
    """Write a two-node graph, with its members' images in the step log beside it."""
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
                        "observations": ["000000", "000002"],
                    },
                    {
                        "id": "s1_search",
                        "page_description": "Search results",
                        "activity": "com.demo.Search",
                        "observations": ["000001"],
                    },
                ],
                "edges": [
                    _saved(
                        {
                            "source": "s0_home",
                            "target": "s1_search",
                            "actions": [{"action": "tap", "x": 5, "y": 6}],
                            "instructions": ["open search"],
                            "visit_count": 2,
                        }
                    ),
                    _saved(
                        {
                            "source": "s1_search",
                            "target": "s1_search",
                            "actions": [{"action": "type", "text": "cafe"}],
                            "schema_deltas": [{"query": {"before": None, "after": "cafe"}}],
                        }
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    for observation_id in ("000000", "000001", "000002"):
        _write_step_image(tmp_path, observation_id, f"member {observation_id}".encode())
    return graph_path


def _write_step_image(tmp_path: Path, observation_id: str, image: bytes) -> str:
    """Write an observation's image into the step log beside the graph; return its data URI."""
    steps = tmp_path / "steps"
    steps.mkdir(exist_ok=True)
    (steps / f"{observation_id}.webp").write_bytes(image)
    return f"data:image/webp;base64,{base64.b64encode(image).decode('ascii')}"


def test_load_graph_reads_nodes_edges_and_screenshots(tmp_path: Path) -> None:
    """A node's member images are its members' WebP images from the step log."""
    graph_path = _write_graph(tmp_path)
    home_uri = _write_step_image(tmp_path, "000000", b"home")
    search_uri = _write_step_image(tmp_path, "000001", b"search")
    later_home_uri = _write_step_image(tmp_path, "000002", b"later home")

    data, graph = load_result = plot.load_graph(str(graph_path))
    assert len(load_result) == 2
    assert data["nodes"][0]["id"] == "s0_home"
    assert set(graph) == {"s0_home", "s1_search"}
    assert graph.nodes["s0_home"]["elements_explored"] == 1
    assert graph.nodes["s0_home"]["elements_total"] == 2
    assert graph.nodes["s0_home"]["member_image_uris"] == [home_uri, later_home_uri]
    assert graph.nodes["s1_search"]["member_image_uris"] == [search_uri]
    assert graph.edges["s0_home", "s1_search"]["label"].startswith("open search")
    # The self-loop edge falls back to an action summary plus its schema delta.
    self_loop_label = graph.edges["s1_search", "s1_search"]["label"]
    assert self_loop_label.startswith("type 'cafe'")
    assert "query" in self_loop_label


_MEMBER = {"observations": ["000000"]}


def test_load_graph_labels_skip_the_placeholders_of_unrecorded_values(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "s0_list", "page_description": "Shopping list", **_MEMBER},
                    {"id": "s1_info", "page_description": "Item info", **_MEMBER},
                ],
                "edges": [
                    _saved(
                        {
                            "source": "s0_list",
                            "target": "s0_list",
                            "actions": [
                                {"action": "swipe"},
                                {"action": "back"},
                                {"action": "type", "text": "milk"},
                            ],
                            "instructions": ["Scroll down the list", "", "Type 'milk'"],
                            "schema_deltas": [
                                {"first_visible_item": {"before": "A", "after": "K"}},
                                None,
                                {"search_query": {"before": None, "after": "milk"}},
                            ],
                        }
                    ),
                    _saved(
                        {
                            "source": "s0_list",
                            "target": "s1_info",
                            "actions": [{"action": "tap", "x": 9, "y": 40}],
                        }
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_step_image(tmp_path, "000000", b"image")

    _data, graph = plot.load_graph(str(graph_path))

    assert graph.edges["s0_list", "s0_list"]["label"] == (
        "Scroll down the list\nType 'milk'\n"
        "[Δ first_visible_item: A → K, search_query: None → milk]\n[1 step]"
    )
    assert graph.edges["s0_list", "s1_info"]["label"] == "tap (9, 40)\n[1 step]"


@pytest.mark.parametrize(
    ("edge", "label"),
    [
        (
            {
                "target": "s1_results",
                "instruction_templates": [{"template": "Search for {query}"}],
                "instructions": ["Search for cafe"],
            },
            "Search for {query}\n[1 step]",
        ),
        (
            {
                "target": "s0_search",
                "actions": [{"action": "back"}, {"action": "back"}],
                "instructions": ["Go back", ""],
                "schema_deltas": [None, {}],
                "num_steps": [2, 3],
            },
            "Go back\n[2 steps]",
        ),
    ],
)
def test_load_graph_labels_prefer_the_template_and_omit_empty_deltas(
    edge: dict[str, object], label: str, tmp_path: Path
) -> None:
    graph_path = tmp_path / "demo.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "s0_search", "page_description": "Search", **_MEMBER},
                    {"id": "s1_results", "page_description": "Results", **_MEMBER},
                ],
                "edges": [
                    _saved(
                        {
                            "source": "s0_search",
                            "actions": [{"action": "tap", "x": 1, "y": 2}],
                            **edge,
                        }
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_step_image(tmp_path, "000000", b"image")

    _data, graph = plot.load_graph(str(graph_path))

    assert graph.edges["s0_search", edge["target"]]["label"] == label


def test_load_graph_reads_an_audited_graph_from_the_shared_step_log(tmp_path: Path) -> None:
    graph_path = _write_graph(tmp_path, stem="demo_audited")
    home_uri = _write_step_image(tmp_path, "000000", b"home")

    _data, graph = plot.load_graph(str(graph_path))
    assert graph.nodes["s0_home"]["member_image_uris"][0] == home_uri


def _write_member_graph(tmp_path: Path, observations: list[str]) -> Path:
    """Write a one-node graph whose node lists ``observations``."""
    graph_path = tmp_path / "demo.json"
    node = {"id": "s0_home", "page_description": "Home", "observations": observations}
    graph_path.write_text(json.dumps({"nodes": [node], "edges": []}), encoding="utf-8")
    return graph_path


def _steps(count: int) -> list[str]:
    return [f"{step:06d}" for step in range(count)]


def test_load_graph_loads_the_ten_oldest_member_images_and_counts_every_member(
    tmp_path: Path,
) -> None:
    observations = _steps(12)
    graph_path = _write_member_graph(tmp_path, observations)
    uris = [_write_step_image(tmp_path, step, f"member {step}".encode()) for step in observations]

    _data, graph = plot.load_graph(str(graph_path))

    assert graph.nodes["s0_home"]["member_image_uris"] == uris[:10]
    assert graph.nodes["s0_home"]["member_total"] == 12


def test_load_graph_rejects_a_member_without_an_image(tmp_path: Path) -> None:
    """Any member counts, not only the ten the panel shows."""
    observations = _steps(12)
    graph_path = _write_member_graph(tmp_path, observations)
    for step in observations:
        if step not in {"000003", "000011"}:
            _write_step_image(tmp_path, step, f"member {step}".encode())

    with pytest.raises(ValueError, match="without an image") as excinfo:
        plot.load_graph(str(graph_path))

    message = str(excinfo.value)
    assert "['000003', '000011']" in message
    assert str(graph_path) in message
    assert "Explore the app again into a new graph directory" in message


@pytest.mark.parametrize(
    ("edge", "problem"),
    [
        ({}, "edge(s) without actions"),
        ({"actions": []}, "edge(s) without actions"),
        (
            _saved({"actions": [{"action": "tap"}], "setups": []}),
            "edge(s) without one entry per action",
        ),
    ],
    ids=["no-actions", "empty-actions", "short-list"],
)
def test_load_graph_rejects_an_edge_without_actions_or_one_entry_per_action(
    tmp_path: Path, edge: dict[str, object], problem: str
) -> None:
    graph_path = tmp_path / "demo.json"
    nodes = [{"id": "s0_home", "page_description": "Home", **_MEMBER}]
    edges = [{"source": "s0_home", "target": "s0_home", **edge}]
    graph_path.write_text(json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8")
    _write_step_image(tmp_path, "000000", b"image")

    with pytest.raises(ValueError, match=re.escape(problem)) as excinfo:
        plot.load_graph(str(graph_path))

    message = str(excinfo.value)
    assert "['s0_home -> s0_home']" in message
    assert str(graph_path) in message
    assert "Explore the app again into a new graph directory" in message


def test_load_graph_rejects_a_file_without_an_edges_list(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    nodes = [{"id": "s0_home", "page_description": "Home", **_MEMBER}]
    graph_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    _write_step_image(tmp_path, "000000", b"image")

    with pytest.raises(ValueError, match="list fields 'nodes' and 'edges'") as excinfo:
        plot.load_graph(str(graph_path))

    assert str(graph_path) in str(excinfo.value)
    assert "Explore the app again into a new graph directory" in str(excinfo.value)


@pytest.mark.parametrize(
    "membership",
    [{}, {"observations": []}, {"observations": "000000"}, {"observations": [7, "000000"]}],
    ids=["no-members", "empty-members", "members-not-a-list", "a-member-not-a-string"],
)
def test_load_graph_rejects_a_node_that_does_not_list_its_members(
    tmp_path: Path, membership: dict[str, object]
) -> None:
    graph_path = tmp_path / "demo.json"
    nodes = [
        {"id": "s0_home", "page_description": "Home", **_MEMBER},
        {"id": "s1_search", "page_description": "Search", **membership},
    ]
    graph_path.write_text(json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="no member observations") as excinfo:
        plot.load_graph(str(graph_path))

    message = str(excinfo.value)
    assert "['s1_search']" in message
    assert str(graph_path) in message
    assert "Explore the app again into a new graph directory" in message


def _vis_nodes(html: str) -> dict[str, dict[str, Any]]:
    """Return the nodes pyvis wrote into the view's ``vis.DataSet``, by ID."""
    opening = "nodes = new vis.DataSet("
    items, _end = json.JSONDecoder().raw_decode(html, html.index(opening) + len(opening))
    return {item["id"]: item for item in items}


def _plot_member_graph(tmp_path: Path, observations: list[str]) -> str:
    """Plot the graph ``_write_member_graph`` writes; return the view's HTML."""
    graph_path = _write_member_graph(tmp_path, observations)
    _data, graph = plot.load_graph(str(graph_path))
    output = tmp_path / "graph.html"
    plot.plot_pyvis(graph, str(output))
    return output.read_text(encoding="utf-8")


def test_plot_pyvis_embeds_the_ten_oldest_member_images_and_states_the_total(
    tmp_path: Path,
) -> None:
    observations = _steps(12)
    uris = [_write_step_image(tmp_path, step, f"member {step}".encode()) for step in observations]

    html = _plot_member_graph(tmp_path, observations)

    node = _vis_nodes(html)["s0_home"]
    assert node["memberImages"] == uris[:10]
    assert node["memberTotal"] == 12
    assert node["label"] == "Home\n[12 screenshots]"
    assert [html.count(uri) for uri in uris] == [1] * 10 + [0, 0]


def test_plot_pyvis_tooltip_shows_the_earliest_member_from_the_panel_s_copy_of_its_image(
    tmp_path: Path,
) -> None:
    """The thumbnail is the earliest member on every render, the first of the panel's
    images, so the page embeds it once."""
    observations = ["000003", "000005", "000008"]
    uris = [_write_step_image(tmp_path, step, f"member {step}".encode()) for step in observations]

    html = _plot_member_graph(tmp_path, observations)

    node = _vis_nodes(html)["s0_home"]
    assert "title" not in node
    assert node["memberImages"] == uris
    tooltip = html.index("function tooltipOf(node)")
    assert (
        "image.src = node.memberImages[0];" in html[tooltip : html.index("nodes.update(", tooltip)]
    )
    assert [html.count(uri) for uri in uris] == [1, 1, 1]


def test_plot_pyvis_gives_each_node_its_tooltip_as_rows_of_text(tmp_path: Path) -> None:
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "graph.html"

    plot.plot_pyvis(graph, str(output))

    nodes = _vis_nodes(output.read_text(encoding="utf-8"))
    assert nodes["s0_home"]["tooltipRows"] == [
        ["Description", "Home"],
        ["Activity", "com.demo.Home"],
        ["Visits", "3"],
        ["State keys", "tab"],
        ["Elements", "1/2 explored"],
        ["Unexplored", "?"],
    ]
    assert nodes["s1_search"]["tooltipRows"] == [
        ["Description", "Search results"],
        ["Activity", "com.demo.Search"],
        ["Visits", "0"],
        ["State keys", "none"],
        ["Elements", "0/0 explored"],
    ]


def test_plot_pyvis_passes_markup_in_a_description_to_the_page_only_as_data(
    tmp_path: Path,
) -> None:
    """Descriptions come from a model: the page holds them as escaped JSON strings, and the
    tooltip script sets them with textContent, so no value is ever parsed as HTML."""
    description = '<script>alert("x")</script> & <b>bold</b>'
    graph = nx.DiGraph()
    graph.add_node("s0_home", page_description=description, elements=[{"description": description}])
    output = tmp_path / "graph.html"

    plot.plot_pyvis(graph, str(output))

    html = output.read_text(encoding="utf-8")
    rows = _vis_nodes(html)["s0_home"]["tooltipRows"]
    assert rows[0] == ["Description", description]
    assert rows[-1] == ["Unexplored", description]
    assert "<script>alert" not in html
    injected = html[html.index("drawGraph();") :]
    tooltip_script = injected[: injected.index('<div id="member-panel"')]
    assert "value.textContent = row[1];" in tooltip_script
    assert not any(
        sink in injected
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")
    )


@pytest.mark.parametrize(
    ("total", "label"),
    [(0, "Home\n[0 screenshots]"), (1, "Home\n[1 screenshot]"), (2, "Home\n[2 screenshots]")],
)
def test_plot_pyvis_labels_each_node_with_its_screenshot_count(
    tmp_path: Path, total: int, label: str
) -> None:
    graph = nx.DiGraph()
    graph.add_node("s0_home", page_description="Home", member_total=total)
    output = tmp_path / "graph.html"

    plot.plot_pyvis(graph, str(output))

    assert _vis_nodes(output.read_text(encoding="utf-8"))["s0_home"]["label"] == label


def test_plot_pyvis_adds_the_tooltips_and_the_member_panel_after_the_graph_is_drawn(
    tmp_path: Path,
) -> None:
    """Both scripts read the network and nodes that the template's drawGraph() call creates."""
    graph = nx.DiGraph()
    graph.add_node("s0_home", page_description="Home")
    output = tmp_path / "graph.html"

    plot.plot_pyvis(graph, str(output))

    html = output.read_text(encoding="utf-8")
    body = html[html.index("<body>") : html.index("</body>")]
    injected = body[body.index("drawGraph();") :]
    assert body.count('<div id="member-panel" hidden>') == 1
    assert "title: tooltipOf(node)" in injected
    assert 'network.on("click"' in injected
    assert body.endswith("})();\n</script>\n")


def test_plot_pyvis_writes_fullscreen_html(tmp_path: Path) -> None:
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "graph.html"
    plot.plot_pyvis(graph, str(output))
    html = output.read_text(encoding="utf-8")
    assert "#mynetwork" in html
    assert "100vh" in html


def test_plot_pyvis_lists_ten_unexplored_elements_and_counts_the_rest(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_node(
        "s0_home",
        elements=[
            {"description": f"Button {letter}", "explored": False} for letter in "ABCDEFGHIJK"
        ],
    )
    output = tmp_path / "graph.html"
    plot.plot_pyvis(graph, str(output))
    html = output.read_text(encoding="utf-8")
    listed = ", ".join(f"Button {letter}" for letter in "ABCDEFGHIJ")
    assert _vis_nodes(html)["s0_home"]["tooltipRows"][-1] == ["Unexplored", f"{listed} (+1 more)"]
    assert "Button K" not in html


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            [],
            {
                "max_node_words": 5,
                "max_edge_words": 5,
                "node_size": 34,
                "node_font_size": 20,
                "edge_font_size": 14,
                "edge_width": 2.6,
            },
        ),
        (
            [
                "--max-node-words",
                "3",
                "--max-edge-words",
                "4",
                "--node-size",
                "40",
                "--node-font-size",
                "18",
                "--edge-font-size",
                "12",
                "--edge-width",
                "1.5",
            ],
            {
                "max_node_words": 3,
                "max_edge_words": 4,
                "node_size": 40,
                "node_font_size": 18,
                "edge_font_size": 12,
                "edge_width": 1.5,
            },
        ),
    ],
)
def test_plot_main_passes_the_pyvis_options_to_the_renderer(
    options: list[str],
    expected: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI's own defaults arrive too, such as 5 edge words where plot_pyvis defaults to 6."""
    received: list[tuple[str, dict[str, object]]] = []

    def record(_graph: nx.DiGraph, output_path: str, *, style: plot.PyvisStyle) -> None:
        received.append((output_path, dataclasses.asdict(style)))

    monkeypatch.setattr(plot, "plot_pyvis", record)
    monkeypatch.setattr(plot, "setup_logging", lambda **_kwargs: None)
    graph_path = _write_graph(tmp_path)

    assert plot.main([str(graph_path), "--backend", "pyvis", "--no-open", *options]) == 0

    assert received == [(str(graph_path.with_suffix(".html")), expected)]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            [],
            {
                "max_node_words": 5,
                "max_edge_words": 5,
                "layout": "circular",
                "show_edge_labels": True,
            },
        ),
        (
            [
                "--max-node-words",
                "2",
                "--max-edge-words",
                "3",
                "--paper-layout",
                "kamada",
                "--hide-edge-labels",
            ],
            {
                "max_node_words": 2,
                "max_edge_words": 3,
                "layout": "kamada",
                "show_edge_labels": False,
            },
        ),
    ],
)
def test_plot_main_passes_the_paper_options_to_the_renderer(
    options: list[str],
    expected: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, dict[str, object]]] = []

    def record(_graph: nx.DiGraph, output_path: str, *, style: plot.PaperStyle) -> None:
        received.append((output_path, dataclasses.asdict(style)))

    monkeypatch.setattr(plot, "plot_paper_graphviz", record)
    monkeypatch.setattr(plot, "setup_logging", lambda **_kwargs: None)
    graph_path = _write_graph(tmp_path)

    assert plot.main([str(graph_path), "--backend", "paper", "--no-open", *options]) == 0

    assert received == [(str(graph_path.with_suffix(".png")), expected)]


def test_plot_matplotlib_writes_an_image(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("agg")
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "graph.png"
    plot.plot_matplotlib(graph, str(output))
    assert output.stat().st_size > 0


def test_plot_paper_matplotlib_writes_an_image(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("agg")
    _data, graph = plot.load_graph(str(_write_graph(tmp_path)))
    output = tmp_path / "paper.png"
    plot.plot_paper_matplotlib(graph, str(output), style=plot.PaperStyle(layout="spring"))
    assert output.stat().st_size > 0


def test_plot_paper_graphviz_falls_back_without_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib as mpl

    mpl.use("agg")
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

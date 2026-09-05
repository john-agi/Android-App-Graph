"""Visualize an Android-App-Graph exploration graph as an interactive HTML or static image.

The plotting libraries live in the optional ``viz`` extra, so every import of
``matplotlib`` and ``pyvis`` is local to the renderer that needs it: ``--help``
and argument validation work without the extra installed.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import math
import re
import shutil
import subprocess
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from android_app_graph.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_VIZ_HINT = (
    "app-graph-plot needs the optional plotting dependencies. "
    "Install them with `uv sync --extra viz` in this checkout "
    "or `pip install 'android-app-graph[viz]'` elsewhere."
)


def _require_viz() -> None:
    """Import the plotting libraries, or raise ImportError with install guidance."""
    try:
        importlib.import_module("pyvis.network")
        importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        msg = f"{_VIZ_HINT} (missing module: {exc.name})"
        raise ImportError(msg) from exc


def _as_digraph(graph: object) -> nx.DiGraph:
    """Return a ``networkx`` result as ``DiGraph``.

    ``networkx`` ships no annotations, so ``copy()`` and ``subgraph()`` come
    back as ``Unknown``. This is the one place that says the graph type is
    preserved, mirroring ``graph_manager._node_id``.
    """
    if isinstance(graph, nx.DiGraph):
        return graph
    msg = f"expected a networkx DiGraph, got {type(graph).__name__}"
    raise TypeError(msg)


def _load_screenshot_b64(screenshots_dir: Path, node_id: str) -> str | None:
    """Load a node's reference screenshot as a base64 data URI."""
    img_path = screenshots_dir / f"{node_id}.png"
    if img_path.exists():
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return None


def load_graph(graph_path: str) -> tuple[dict[str, Any], nx.DiGraph]:
    """Load an Android-App-Graph JSON graph and return (raw_data, nx.DiGraph)."""
    path = Path(graph_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    screenshots_dir = path.parent / (path.stem + "_screenshots")
    if not screenshots_dir.exists():
        # An audited graph reuses the screenshots of the graph it was derived from.
        base_stem = path.stem.removesuffix("_audited")
        screenshots_dir = path.parent / (base_stem + "_screenshots")

    G = nx.DiGraph()

    for node in data["nodes"]:
        screenshot_uri = _load_screenshot_b64(screenshots_dir, node["id"])
        elements = node.get("interactable_elements", [])
        n_explored = sum(1 for e in elements if e.get("explored", False))
        G.add_node(
            node["id"],
            page_description=node["page_description"],
            activity=node.get("activity", ""),
            visit_count=node.get("visit_count", 0),
            num_keys=len(node.get("state_schema", {})),
            state_keys=list(node.get("state_schema", {}).keys()),
            elements=elements,
            elements_explored=n_explored,
            elements_total=len(elements),
            screenshot_uri=screenshot_uri,
        )

    for edge in data["edges"]:
        templates = edge.get("instruction_templates", [])
        instructions = edge.get("instructions", [])

        if templates and isinstance(templates[0], dict) and templates[0].get("template"):
            label = templates[0]["template"]
        elif instructions:
            label = "\n".join(instructions)
        else:
            label = _summarize_actions(edge["actions"])

        is_self_loop = edge["source"] == edge["target"]
        schema_deltas = edge.get("schema_deltas", [])
        delta_str = ""
        if is_self_loop and schema_deltas:
            delta_parts = []
            for delta in schema_deltas:
                if delta:
                    for k, v in delta.items():
                        delta_parts.append(f"{k}: {v.get('before')} → {v.get('after')}")
            if delta_parts:
                delta_str = "\n[Δ " + ", ".join(delta_parts) + "]"

        num_steps_list = edge.get("num_steps", [])
        weight = min(num_steps_list) if num_steps_list else len(edge["actions"])
        weight_str = f"\n[{weight} step{'s' if weight != 1 else ''}]"

        G.add_edge(
            edge["source"],
            edge["target"],
            label=label + delta_str + weight_str,
            visit_count=edge.get("visit_count", 0),
            num_actions=len(edge["actions"]),
            num_steps=weight,
            is_self_loop=is_self_loop,
        )

    return data, G


def _summarize_actions(actions: list[Any]) -> str:
    """Create a readable label from action list."""
    if not actions:
        return "?"

    def _describe_action(a: dict[str, Any]) -> str:
        act = str(a.get("action", "?"))
        if act == "tap":
            return f"tap ({a.get('x')}, {a.get('y')})"
        if act == "type":
            return f"type '{a.get('text', '')}'"
        if act == "swipe":
            return f"swipe ({a.get('x1')},{a.get('y1')})→({a.get('x2')},{a.get('y2')})"
        return act

    # An entry may be a whole action sequence rather than a single action.
    first = actions[0]
    if isinstance(first, list):
        return " → ".join(_describe_action(a) for a in first)
    return _describe_action(first)


def _shorten_words(text: str | None, max_words: int = 6, max_chars: int = 42) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "..."
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _shorten_edge_label(text: str | None, max_words: int = 8, max_chars: int = 52) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    text = re.sub(r"\{\{[^}]+\}\}", "value", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(
        r"\b(?:the|a|an|to|on|in|with|button|icon|option)\b", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    return _shorten_words(text, max_words=max_words, max_chars=max_chars)


def _wrap_label(text: str | None, width: int = 18) -> str:
    words = str(text or "").split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\\n".join(lines)


def _node_category(desc: str) -> str:
    desc_l = desc.lower()
    if any(k in desc_l for k in ("search", "results", "list")):
        return "search"
    if any(k in desc_l for k in ("settings", "filter", "sort", "menu", "dialog", "sheet")):
        return "settings"
    if any(k in desc_l for k in ("detail", "profile", "article", "video", "reader")):
        return "detail"
    if any(k in desc_l for k in ("home", "main", "map", "feed", "library")):
        return "home"
    return "other"


def _largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy containing only the largest weakly connected component."""
    if G.number_of_nodes() == 0:
        return _as_digraph(G.copy())
    components = list(nx.weakly_connected_components(G))
    if not components:
        return _as_digraph(G.copy())
    largest = max(components, key=len)
    return _as_digraph(G.subgraph(largest).copy())


def _drop_external_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """Remove launcher/permission/external-app nodes that stretch paper layouts."""
    keep = []
    for node, data in G.nodes(data=True):
        text = f"{node} {data.get('page_description', '')}".lower()
        if node.startswith("ext_") or "external app" in text:
            continue
        keep.append(node)
    return _as_digraph(G.subgraph(keep).copy())


def _trim_leaf_rounds(G: nx.DiGraph, rounds: int) -> nx.DiGraph:
    """Iteratively remove degree-0/1 leaves from a copy of the graph."""
    H = _as_digraph(G.copy())
    for _ in range(max(0, rounds)):
        leaves = [node for node in H.nodes() if H.in_degree(node) + H.out_degree(node) <= 1]
        if not leaves or len(leaves) == H.number_of_nodes():
            break
        H.remove_nodes_from(leaves)
    return H


def _prepare_paper_graph(
    G: nx.DiGraph,
    *,
    keep_disconnected: bool,
    keep_external: bool,
    trim_leaves: int,
) -> nx.DiGraph:
    """Apply paper-figure filters and report what changed."""
    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()
    H = _as_digraph(G.copy())

    if not keep_external:
        H = _drop_external_nodes(H)
    if not keep_disconnected:
        H = _largest_weak_component(H)
    if trim_leaves:
        H = _trim_leaf_rounds(H, trim_leaves)
        if not keep_disconnected:
            H = _largest_weak_component(H)

    removed_nodes = original_nodes - H.number_of_nodes()
    removed_edges = original_edges - H.number_of_edges()
    if removed_nodes or removed_edges:
        logger.info(
            "Paper view: kept %d nodes, %d edges; removed %d nodes and %d edges.",
            H.number_of_nodes(),
            H.number_of_edges(),
            removed_nodes,
            removed_edges,
        )
    return H


def plot_paper_graphviz(
    G: nx.DiGraph,
    output_path: str,
    *,
    max_node_words: int = 5,
    max_edge_words: int = 5,
    layout: str = "circular",
    show_edge_labels: bool = True,
) -> None:
    """Create a paper-friendly static graph using Graphviz."""
    dot_bin = shutil.which("dot")
    if not dot_bin:
        logger.warning("Graphviz 'dot' command not found; using matplotlib paper fallback.")
        plot_paper_matplotlib(
            G,
            output_path,
            max_node_words=max_node_words,
            max_edge_words=max_edge_words,
            layout=layout,
            show_edge_labels=show_edge_labels,
        )
        return

    def q(value: str) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def attrs(items: dict[str, str]) -> str:
        return ", ".join(f"{key}={q(value)}" for key, value in items.items())

    palette = {
        "home": ("#d9f0ff", "#2563eb"),
        "search": ("#dcfce7", "#16a34a"),
        "settings": ("#fef3c7", "#d97706"),
        "detail": ("#fce7f3", "#db2777"),
        "other": ("#eef2ff", "#4f46e5"),
    }

    lines = [
        "digraph G {",
        (
            '  graph [rankdir="LR", bgcolor="white", pad="0.35", nodesep="0.55", '
            'ranksep="0.85", splines="spline", overlap="false", outputorder="edgesfirst", '
            'fontname="Helvetica", fontsize="18", labelloc="t", '
            f"label={q(f'Android-App-Graph Runtime Graph\\n{G.number_of_nodes()} states, {G.number_of_edges()} transitions')}];"
        ),
        (
            '  node [shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10", '
            'margin="0.10,0.07", penwidth="1.4", color="#2f3a4a"];'
        ),
        (
            '  edge [fontname="Helvetica", fontsize="8", color="#758195", '
            'fontcolor="#475467", arrowsize="0.55", penwidth="1.15"];'
        ),
    ]

    for node_id, data in G.nodes(data=True):
        desc = data.get("page_description", node_id)
        category = _node_category(desc)
        fill, border = palette[category]
        visits = data.get("visit_count", 0)
        short_desc = _wrap_label(_shorten_words(desc, max_node_words, 38), width=16)
        node_attrs = {
            "label": f"{node_id}\\n{short_desc}",
            "fillcolor": fill,
            "color": border,
            "penwidth": f"{1.2 + min(visits, 5) * 0.18:.2f}",
        }
        lines.append(f"  {q(node_id)} [{attrs(node_attrs)}];")

    edge_seen: Counter[tuple[str, str]] = Counter()
    for src, tgt, data in G.edges(data=True):
        data = G.edges[src, tgt]
        label = data.get("label", "")
        short = _wrap_label(_shorten_words(label, max_edge_words, 34), width=14)
        visits = data.get("visit_count", 0)
        edge_attrs = {
            "label": f" {short} " if short else "",
            "penwidth": f"{1.0 + math.log1p(max(visits, 0)) * 0.45:.2f}",
        }
        if src == tgt or data.get("is_self_loop"):
            edge_attrs.update(
                {
                    "style": "dashed",
                    "color": "#9ca3af",
                    "fontcolor": "#6b7280",
                }
            )
        else:
            edge_seen[(src, tgt)] += 1
            if edge_seen[(src, tgt)] > 1:
                edge_attrs["color"] = "#94a3b8"

        lines.append(f"  {q(src)} -> {q(tgt)} [{attrs(edge_attrs)}];")

    lines.append("}")
    dot_path = Path(output_path).with_suffix(".dot")
    dot_path.write_text("\n".join(lines), encoding="utf-8")

    ext = Path(output_path).suffix.lstrip(".").lower() or "png"
    fmt = "pdf" if ext == "pdf" else "svg" if ext == "svg" else "png"
    subprocess.run(
        [dot_bin, f"-T{fmt}", str(dot_path), "-o", output_path],
        check=True,
    )
    logger.info("Paper graph saved to %s", output_path)


def plot_paper_matplotlib(
    G: nx.DiGraph,
    output_path: str,
    *,
    max_node_words: int = 5,
    max_edge_words: int = 5,
    layout: str = "circular",
    show_edge_labels: bool = True,
) -> None:
    """Paper-style fallback renderer that does not require Graphviz."""
    import matplotlib.pyplot as plt

    n_nodes = max(G.number_of_nodes(), 1)
    fig_w = min(34, max(16, n_nodes * 0.34))
    fig_h = min(24, max(11, n_nodes * 0.24))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)

    if layout == "circular":
        pos = nx.circular_layout(G, scale=1.0)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G, scale=1.0)
    else:
        pos = nx.spring_layout(
            G,
            k=5.0 / math.sqrt(n_nodes),
            iterations=500,
            seed=7,
            scale=1.0,
        )

    palette = {
        "home": "#d9f0ff",
        "search": "#dcfce7",
        "settings": "#fef3c7",
        "detail": "#fce7f3",
        "other": "#eef2ff",
    }
    borders = {
        "home": "#2563eb",
        "search": "#16a34a",
        "settings": "#d97706",
        "detail": "#db2777",
        "other": "#4f46e5",
    }
    categories = {
        node: _node_category(data.get("page_description", node))
        for node, data in G.nodes(data=True)
    }

    for category, node_color in palette.items():
        nodes = [node for node, cat in categories.items() if cat == category]
        if not nodes:
            continue
        visits = [G.nodes[node].get("visit_count", 0) for node in nodes]
        sizes = [720 + min(v, 8) * 70 for v in visits]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=sizes,
            node_color=node_color,
            edgecolors=borders[category],
            linewidths=1.4,
            ax=ax,
        )

    non_self_edges = [(s, t) for s, t in G.edges() if s != t]
    self_edges = [(s, t) for s, t in G.edges() if s == t]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=non_self_edges,
        edge_color="#94a3b8",
        width=[0.8 + math.log1p(G.edges[e].get("visit_count", 0)) * 0.35 for e in non_self_edges],
        arrows=True,
        arrowsize=9,
        connectionstyle="arc3,rad=0.10",
        ax=ax,
    )
    if self_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=self_edges,
            edge_color="#9ca3af",
            style="dashed",
            arrows=True,
            arrowsize=8,
            connectionstyle="arc3,rad=0.28",
            ax=ax,
        )

    node_labels = {
        node: f"{node}\n{_shorten_words(data.get('page_description', node), max_node_words, 34)}"
        for node, data in G.nodes(data=True)
    }
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=5.5,
        font_family="DejaVu Sans",
        ax=ax,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.68,
        },
    )

    if show_edge_labels:
        edge_labels = {
            (s, t): _shorten_words(data.get("label", ""), max_edge_words, 24)
            for s, t, data in G.edges(data=True)
            if data.get("label") and s != t
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=3.7,
            font_color="#475467",
            rotate=False,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.62,
            },
            ax=ax,
        )

    ax.set_title(
        f"Android-App-Graph Runtime Graph: {G.number_of_nodes()} states, {G.number_of_edges()} transitions",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Paper graph saved to %s", output_path)


def plot_pyvis(
    G: nx.DiGraph,
    output_path: str,
    *,
    max_node_words: int = 5,
    max_edge_words: int = 6,
    node_size: int = 34,
    node_font_size: int = 20,
    edge_font_size: int = 14,
    edge_width: float = 2.6,
) -> None:
    """Create an interactive HTML graph using pyvis."""
    from pyvis.network import Network

    net = Network(
        height="100vh",
        width="100vw",
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Spread nodes far apart: these graphs are dense and the labels are long.
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.15,
          "springLength": 400,
          "springConstant": 0.01,
          "damping": 0.3,
          "avoidOverlap": 1
        },
        "solver": "barnesHut",
        "stabilization": {"iterations": 500}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "smooth": {"type": "curvedCW", "roundness": 0.15},
        "font": {"size": 14, "align": "horizontal", "background": "rgba(255,255,255,0.9)"},
        "color": {"color": "#5c6bc0", "highlight": "#1a237e", "opacity": 0.7},
        "width": 2.6
      },
      "nodes": {
        "font": {"size": 20, "face": "Arial"},
        "shape": "box",
        "margin": 16,
        "borderWidth": 3,
        "shadow": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    palette = [
        "#e3f2fd",
        "#e8f5e9",
        "#fff3e0",
        "#fce4ec",
        "#f3e5f5",
        "#e0f7fa",
        "#fff9c4",
        "#efebe9",
        "#e8eaf6",
        "#f1f8e9",
    ]

    for idx, (node_id, data) in enumerate(G.nodes(data=True)):
        desc = data.get("page_description", node_id)
        short_desc = _shorten_words(desc, max_node_words, 40)
        visits = data.get("visit_count", 0)
        state_keys = data.get("state_keys", [])
        screenshot_uri = data.get("screenshot_uri")
        bg = palette[idx % len(palette)]

        keys_str = ", ".join(state_keys) if state_keys else "none"
        elements = data.get("elements", [])
        n_explored = data.get("elements_explored", 0)
        n_total = data.get("elements_total", 0)
        tooltip = (
            f"<b>{node_id}</b><br>"
            f"Activity: {data.get('activity', '?')}<br>"
            f"Visits: {visits}<br>"
            f"State keys: {keys_str}<br>"
            f"Elements: {n_explored}/{n_total} explored"
        )
        unexplored = [e for e in elements if not e.get("explored", False)]
        if unexplored:
            tooltip += "<br><b>Unexplored:</b> "
            tooltip += ", ".join(e.get("description", "?") for e in unexplored[:10])
            if len(unexplored) > 10:
                tooltip += f" (+{len(unexplored) - 10} more)"
        if screenshot_uri:
            tooltip += (
                f'<br><br><img src="{screenshot_uri}" '
                f'style="max-width:250px; max-height:450px; border:1px solid #ccc; border-radius:4px;">'
            )

        net.add_node(
            node_id,
            label=short_desc,
            title=tooltip,
            color={
                "background": bg,
                "border": "#5c6bc0",
                "highlight": {"background": "#c5cae9", "border": "#1a237e"},
            },
            size=node_size,
            font={"size": node_font_size, "face": "Arial"},
            borderWidth=3,
        )

    edge_counts: Counter[tuple[str, str]] = Counter()

    for src, tgt, data in G.edges(data=True):
        label = data.get("label", "")
        visits = data.get("visit_count", 0)
        n_actions = data.get("num_actions", 0)

        tooltip = f"Visits: {visits}<br>Action sequences: {n_actions}<br>{label}"
        short_label = _shorten_edge_label(label, max_edge_words, 54)

        # Vary curvature for parallel edges
        pair = (min(src, tgt), max(src, tgt))
        edge_counts[pair] += 1
        roundness = 0.15 + (edge_counts[pair] - 1) * 0.15

        net.add_edge(
            src,
            tgt,
            label=short_label,
            title=tooltip,
            smooth={"type": "curvedCW", "roundness": roundness},
            width=edge_width,
            font={
                "size": edge_font_size,
                "align": "horizontal",
                "background": "rgba(255,255,255,0.92)",
            },
        )

    net.save_graph(output_path)

    # Inject fullscreen CSS so the graph fills the entire browser viewport.
    html = Path(output_path).read_text(encoding="utf-8")
    fullscreen_css = (
        "<style>html, body { margin: 0; padding: 0; overflow: hidden; "
        "width: 100vw; height: 100vh; } #mynetwork { width: 100vw !important; "
        "height: 100vh !important; }</style>"
    )
    html = html.replace("<head>", f"<head>{fullscreen_css}", 1)
    Path(output_path).write_text(html, encoding="utf-8")

    logger.info("Interactive graph saved to %s", output_path)


def plot_graphviz(G: nx.DiGraph, output_path: str) -> None:
    """Create a static graph image using graphviz."""
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="TB", fontsize="12", bgcolor="white")
    A.node_attr.update(
        shape="box",
        style="filled,rounded",
        fillcolor="lightblue",
        fontsize="11",
    )
    A.edge_attr.update(fontsize="9", fontcolor="gray40")

    for node in A.nodes():
        data = G.nodes[node.name]
        desc = data.get("page_description", node.name)
        visits = data.get("visit_count", 0)
        node.attr["label"] = f"{desc}\\n({visits} visits)"

    for edge in A.edges():
        data = G.edges[edge[0], edge[1]]
        label = data.get("label", "")
        short = label[:35] + "..." if len(label) > 35 else label
        edge.attr["label"] = f"  {short}  "

    A.draw(output_path, prog="dot")
    logger.info("Graph image saved to %s", output_path)


def plot_matplotlib(G: nx.DiGraph, output_path: str) -> None:
    """Fallback: plot with matplotlib + networkx."""
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    labels = {
        n: f"{d.get('page_description', n)}\n({d.get('visit_count', 0)} visits)"
        for n, d in G.nodes(data=True)
    }
    visits = [G.nodes[n].get("visit_count", 1) for n in G.nodes()]
    node_sizes = [800 + v * 200 for v in visits]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color="skyblue", edgecolors="steelblue"
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15",
    )

    edge_labels = {(s, t): d.get("label", "")[:30] for s, t, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

    ax.set_title(
        f"Android-App-Graph Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graph image saved to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-graph-plot",
        description="Plot an Android-App-Graph exploration graph.",
    )
    parser.add_argument("graph", type=str, help="Path to the graph JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: same dir as graph, .html for pyvis)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pyvis",
        choices=["pyvis", "graphviz", "matplotlib", "paper"],
        help="Visualization backend (default: pyvis)",
    )
    parser.add_argument(
        "--max-node-words",
        type=int,
        default=5,
        help="Maximum words shown in each paper node label.",
    )
    parser.add_argument(
        "--max-edge-words",
        type=int,
        default=5,
        help="Maximum words shown in each paper edge label.",
    )
    parser.add_argument(
        "--keep-disconnected",
        action="store_true",
        help="For --backend paper, keep disconnected components instead of showing only the largest component.",
    )
    parser.add_argument(
        "--keep-external",
        action="store_true",
        help="For --backend paper, keep external app/launcher nodes.",
    )
    parser.add_argument(
        "--trim-leaves",
        type=int,
        default=1,
        help="For --backend paper, remove this many rounds of degree-0/1 leaf nodes after filtering.",
    )
    parser.add_argument(
        "--paper-layout",
        choices=["circular", "kamada", "spring"],
        default="circular",
        help="Fallback matplotlib layout for --backend paper.",
    )
    parser.add_argument(
        "--hide-edge-labels",
        action="store_true",
        help="For --backend paper, hide edge labels to reduce clutter.",
    )
    parser.add_argument(
        "--node-size",
        type=int,
        default=34,
        help="For --backend pyvis, fixed node size.",
    )
    parser.add_argument(
        "--node-font-size",
        type=int,
        default=20,
        help="For --backend pyvis, node label font size.",
    )
    parser.add_argument(
        "--edge-font-size",
        type=int,
        default=14,
        help="For --backend pyvis, edge label font size.",
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=2.6,
        help="For --backend pyvis, edge line width.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the result in a web browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _require_viz()
    except ImportError as exc:
        parser.exit(1, f"{exc}\n")

    graph_path = Path(args.graph)
    if not graph_path.exists():
        parser.error(f"graph file not found: {graph_path}")

    setup_logging(level=logging.INFO)

    _data, G = load_graph(str(graph_path))

    if args.output:
        output_path = args.output
    else:
        ext = ".html" if args.backend == "pyvis" else ".png"
        output_path = str(graph_path.with_suffix(ext))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if args.backend == "paper":
        G = _prepare_paper_graph(
            G,
            keep_disconnected=args.keep_disconnected,
            keep_external=args.keep_external,
            trim_leaves=args.trim_leaves,
        )

    if args.backend == "pyvis":
        plot_pyvis(
            G,
            output_path,
            max_node_words=args.max_node_words,
            max_edge_words=args.max_edge_words,
            node_size=args.node_size,
            node_font_size=args.node_font_size,
            edge_font_size=args.edge_font_size,
            edge_width=args.edge_width,
        )
    elif args.backend == "graphviz":
        plot_graphviz(G, output_path)
    elif args.backend == "matplotlib":
        plot_matplotlib(G, output_path)
    elif args.backend == "paper":
        plot_paper_graphviz(
            G,
            output_path,
            max_node_words=args.max_node_words,
            max_edge_words=args.max_edge_words,
            layout=args.paper_layout,
            show_edge_labels=not args.hide_edge_labels,
        )

    if not args.no_open:
        webbrowser.open(Path(output_path).resolve().as_uri())

    return 0

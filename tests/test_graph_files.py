"""Tests for android_app_graph.graph_files.

Graph-file discovery, per-node reference-screenshot lookup, atomic JSON
writing, and graph-structure validation shared by every loader.
"""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from typing import IO

import pytest

from android_app_graph import graph_files


def test_write_json_atomically_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_write_json_atomically_honours_indent_and_ensure_ascii(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"name": "café"}, indent=2, ensure_ascii=False)
    assert path.read_text(encoding="utf-8") == '{\n  "name": "café"\n}'


def test_write_json_atomically_is_atomic_on_a_failed_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-dump must leave the file as either the previous complete
    version or the new one, never a truncated mix of the two, and must not
    leave a stray temporary file behind.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0, 2.0]})
    original = path.read_text(encoding="utf-8")

    def _dump_then_blow_up(_obj: object, fp: IO[str], **_kwargs: object) -> None:
        fp.write('{"n9": [0.0')  # a partial write, as a real crash mid-dump would leave
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(graph_files.json, "dump", _dump_then_blow_up)

    with pytest.raises(ValueError, match="boom"):
        graph_files.write_json_atomically(path, {"n1": [9.9]})

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomically_unlinks_the_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``os.replace`` (EPERM, EBUSY, target is a directory, ...) must not
    orphan the temp file in the target directory: an unguarded replace leaves a
    stray ``data.json.<random>.tmp`` behind, and every later run adds another one.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0, 2.0]})
    original = path.read_text(encoding="utf-8")

    def _raise_replace(_src: object, _dst: object) -> None:
        msg = "Device or resource busy"
        raise OSError(msg)

    monkeypatch.setattr(graph_files.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="Device or resource busy"):
        graph_files.write_json_atomically(path, {"n1": [9.9]})

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomically_gives_a_fresh_file_the_umask_mode(tmp_path: Path) -> None:
    """A fresh file must get exactly the mode ``open(path, "w")`` would give,
    not mkstemp's hardcoded 0600 -- otherwise a graph file written by one user
    (or a CI job) becomes unreadable to another process reading the same
    shared graph directory as a different user.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0]})

    sibling = tmp_path / "sibling.txt"
    with sibling.open("w", encoding="utf-8") as f:
        f.write("x")

    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(sibling.stat().st_mode)


def test_write_json_atomically_preserves_an_existing_files_mode(tmp_path: Path) -> None:
    """A rewrite must keep the file's current mode, matching what in-place
    truncation (the pre-atomic-write behaviour) did, so an operator's chmod on
    a shared graph directory survives a rewrite.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0]})
    path.chmod(0o600)

    graph_files.write_json_atomically(path, {"n1": [2.0]})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_iter_graph_files_without_a_graph_dir(tmp_path: Path) -> None:
    assert graph_files.iter_graph_files(tmp_path / "absent") == []


def test_iter_graph_files_prefers_the_audited_graph(tmp_path: Path) -> None:
    app_dir = tmp_path / "eboox"
    app_dir.mkdir()
    (app_dir / "eboox.json").write_text("{}", encoding="utf-8")
    audited = app_dir / "eboox_audited.json"
    audited.write_text("{}", encoding="utf-8")
    assert graph_files.iter_graph_files(tmp_path) == [("eboox", audited)]


def test_iter_graph_files_sorts_apps_and_skips_side_files(tmp_path: Path) -> None:
    for app in ("zebra", "alpha"):
        app_dir = tmp_path / app
        app_dir.mkdir()
        (app_dir / f"{app}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha" / "alpha_audit_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha" / "alpha.image_emb.json").write_text("{}", encoding="utf-8")
    (tmp_path / "loose.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    assert graph_files.iter_graph_files(tmp_path) == [
        ("alpha", tmp_path / "alpha" / "alpha.json"),
        ("zebra", tmp_path / "zebra" / "zebra.json"),
    ]


def test_reference_screenshot_path_finds_the_node_in_the_app_directory(tmp_path: Path) -> None:
    """Only the app directory exists (a plain, never-audited graph)."""
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n1.png").write_bytes(b"shot")
    graph_path = app_dir / "demo.json"
    assert graph_files.reference_screenshot_path(graph_path, "n1") == app_screenshots / "n1.png"


def test_reference_screenshot_path_prefers_the_stem_directory_for_a_reexplored_node(
    tmp_path: Path,
) -> None:
    """``GraphManager.save_graph`` writes a re-explored node's screenshot into
    ``<stem>_screenshots``; when the node is there it must win over the older
    copy that may still sit under the app-name directory.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n1.png").write_bytes(b"stale")
    stem_screenshots = app_dir / "demo_audited_screenshots"
    stem_screenshots.mkdir()
    (stem_screenshots / "n1.png").write_bytes(b"fresh")
    graph_path = app_dir / "demo_audited.json"
    assert graph_files.reference_screenshot_path(graph_path, "n1") == stem_screenshots / "n1.png"


def test_reference_screenshot_path_falls_back_when_the_node_is_only_in_the_app_directory(
    tmp_path: Path,
) -> None:
    """Both directories exist (some nodes were re-explored, this one was not),
    so a node missing from ``<stem>_screenshots`` must still resolve to its
    screenshot under the app-name directory rather than being reported missing.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n2.png").write_bytes(b"shot")
    (app_dir / "demo_audited_screenshots").mkdir()
    graph_path = app_dir / "demo_audited.json"
    assert graph_files.reference_screenshot_path(graph_path, "n2") == app_screenshots / "n2.png"


def test_reference_screenshot_path_does_not_fall_back_for_a_sibling_graph(
    tmp_path: Path,
) -> None:
    """A sibling graph in the same app directory (an operator's ``demo_v1.json``
    kept next to ``demo.json``) must never borrow another graph's screenshot for
    a colliding node id. Only the plain graph and its audited pair produce the
    split layout the app-directory fallback exists for.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "s0.png").write_bytes(b"shot")
    graph_path = app_dir / "demo_v1.json"
    assert graph_files.reference_screenshot_path(graph_path, "s0") is None


def test_reference_screenshot_path_returns_none_when_neither_directory_has_the_node(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    graph_path = app_dir / "demo_audited.json"
    assert graph_files.reference_screenshot_path(graph_path, "n1") is None


def test_reference_screenshot_b64_reads_and_encodes_the_file(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    screenshots = app_dir / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(b"shot-bytes")
    graph_path = app_dir / "demo.json"
    assert graph_files.reference_screenshot_b64(graph_path, "n1") == base64.b64encode(
        b"shot-bytes"
    ).decode("ascii")


def test_reference_screenshot_b64_is_none_without_a_screenshot(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    graph_path = app_dir / "demo.json"
    assert graph_files.reference_screenshot_b64(graph_path, "n1") is None


def test_iter_graph_files_can_select_one_app(tmp_path: Path) -> None:
    for app in ("demo", "other"):
        app_dir = tmp_path / app
        app_dir.mkdir()
        (app_dir / f"{app}.json").write_text("{}", encoding="utf-8")
    assert graph_files.iter_graph_files(tmp_path, "demo") == [
        ("demo", tmp_path / "demo" / "demo.json")
    ]


def test_iter_graph_files_skips_an_unknown_app(tmp_path: Path) -> None:
    assert graph_files.iter_graph_files(tmp_path, "absent") == []

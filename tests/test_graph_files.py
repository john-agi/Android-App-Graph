"""Tests for android_app_graph.graph_files.

Atomic JSON and byte writing, the step log's layout, model-tagged vector files,
and graph-structure validation.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from typing import IO, Any

import pytest

from android_app_graph import graph_files


def test_write_json_atomically_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_write_json_atomically_fsyncs_the_temp_file_before_the_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "either the previous complete file or the new one" guarantee only
    survives a power loss, not just a process crash, if the temp file's data
    reached disk before the rename -- os.fsync must run once, on the temp
    file's own descriptor, before os.replace runs -- and if the rename itself
    reached disk too, which needs a second os.fsync, on the containing
    directory's descriptor, after os.replace runs.
    """
    path = tmp_path / "data.json"
    calls: list[str] = []
    opened_fds: list[int] = []
    fsynced_fds: list[int] = []

    original_open = graph_files.os.open

    def _recording_open(file: Path, flags: int, mode: int = 0o777) -> int:
        fd = original_open(file, flags, mode)
        opened_fds.append(fd)
        return fd

    original_fsync = graph_files.os.fsync

    def _recording_fsync(fd: int) -> None:
        calls.append("fsync")
        fsynced_fds.append(fd)
        original_fsync(fd)

    original_replace = graph_files.os.replace

    def _recording_replace(src: Path, dst: Path) -> None:
        calls.append("replace")
        original_replace(src, dst)

    monkeypatch.setattr(graph_files.os, "open", _recording_open)
    monkeypatch.setattr(graph_files.os, "fsync", _recording_fsync)
    monkeypatch.setattr(graph_files.os, "replace", _recording_replace)

    graph_files.write_json_atomically(path, {"n1": [1.0]})

    assert calls == ["fsync", "replace", "fsync"]
    assert fsynced_fds == opened_fds


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


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root passes os.access for every file regardless of mode, the same as root's "
    "in-place write succeeding, so a 0o444 target denies nothing to run as root here",
)
def test_write_json_atomically_rejects_a_read_only_target(tmp_path: Path) -> None:
    """``os.replace`` needs only directory write permission, so without this
    guard an operator's ``chmod 444`` freeze on a shared graph directory is
    silently defeated by a rewrite; the guard raises ``PermissionError``
    instead and leaves the file and its directory untouched.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0]})
    path.chmod(0o444)

    with pytest.raises(PermissionError) as excinfo:
        graph_files.write_json_atomically(path, {"n1": [2.0]})

    assert str(path) in str(excinfo.value)
    assert json.loads(path.read_text(encoding="utf-8")) == {"n1": [1.0]}
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomically_rejects_a_read_only_target_by_os_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard itself, exercised independently of the privilege level the
    test happens to run under: real ``os.access`` always passes for root
    regardless of a file's mode (see the skipped chmod-based sibling test
    above), so a container running the suite as root must not lose coverage
    of the guard's own behaviour once ``os.access`` reports it unwritable.
    """
    path = tmp_path / "data.json"
    graph_files.write_json_atomically(path, {"n1": [1.0]})

    monkeypatch.setattr(graph_files.os, "access", lambda *_a, **_kw: False)

    with pytest.raises(PermissionError) as excinfo:
        graph_files.write_json_atomically(path, {"n1": [2.0]})

    assert str(path) in str(excinfo.value)
    assert json.loads(path.read_text(encoding="utf-8")) == {"n1": [1.0]}
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomically_writes_through_a_symlink(tmp_path: Path) -> None:
    """``os.replace`` over a symlink replaces the link itself: with
    ``demo_text_emb.json -> shared_cache.json``, writing must land in
    ``shared_cache.json`` and leave the link a link, not silently detach it
    into a plain file that shadows the (unchanged) shared target.
    """
    target = tmp_path / "shared_cache.json"
    target.write_text('{"old": 1}', encoding="utf-8")
    link = tmp_path / "demo_text_emb.json"
    link.symlink_to(target)

    graph_files.write_json_atomically(link, {"new": 2})

    assert link.is_symlink()
    assert link.resolve() == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": 2}


def test_write_bytes_atomically_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "000000.webp"
    graph_files.write_bytes_atomically(path, b"RIFF....WEBP")
    assert path.read_bytes() == b"RIFF....WEBP"
    assert list(tmp_path.iterdir()) == [path]


def test_write_bytes_atomically_replaces_an_existing_file(tmp_path: Path) -> None:
    """An image left by a crash before its record is overwritten when its number is reused."""
    path = tmp_path / "000000.webp"
    path.write_bytes(b"left by a crash")
    graph_files.write_bytes_atomically(path, b"new image")
    assert path.read_bytes() == b"new image"


def test_write_bytes_atomically_unlinks_the_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``os.replace`` leaves neither the target nor a stray temp file."""

    def _raise_replace(_src: object, _dst: object) -> None:
        msg = "Device or resource busy"
        raise OSError(msg)

    monkeypatch.setattr(graph_files.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="Device or resource busy"):
        graph_files.write_bytes_atomically(tmp_path / "000000.webp", b"image")

    assert list(tmp_path.iterdir()) == []


def test_app_graph_path_is_the_app_directory_named_after_the_app(tmp_path: Path) -> None:
    assert graph_files.app_graph_path(tmp_path, "demo") == tmp_path / "demo" / "demo.json"


def test_an_explored_graph_and_its_audited_copy_share_one_step_log(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo"
    steps = graph_files.graph_steps_dir(app_dir / "demo.json")
    assert steps == app_dir / "steps"
    assert graph_files.graph_steps_dir(app_dir / "demo_audited.json") == steps


def test_an_explored_graph_and_its_audited_copy_keep_their_own_text_vectors(
    tmp_path: Path,
) -> None:
    assert graph_files.text_embeddings_path(tmp_path / "demo.json") == (
        tmp_path / "demo_text_emb.json"
    )
    assert graph_files.text_embeddings_path(tmp_path / "demo_audited.json") == (
        tmp_path / "demo_audited_text_emb.json"
    )


def test_an_explored_graph_and_its_audited_copy_keep_their_own_identify_files(
    tmp_path: Path,
) -> None:
    assert graph_files.identify_file_path(tmp_path / "demo.json") == (
        tmp_path / "demo_identify.jsonl"
    )
    assert graph_files.identify_file_path(tmp_path / "demo_audited.json") == (
        tmp_path / "demo_audited_identify.jsonl"
    )


def test_json_lines_are_appended_and_read_back_in_order(tmp_path: Path) -> None:
    path = tmp_path / "demo_identify.jsonl"
    assert graph_files.read_json_lines(path) == []

    graph_files.append_json_line(path, {"observation": "000001", "text": "Réglages"})
    graph_files.append_json_line(path, {"observation": "000003"})

    assert path.read_text(encoding="utf-8") == (
        '{"observation": "000001", "text": "Réglages"}\n{"observation": "000003"}\n'
    )
    assert graph_files.read_json_lines(path) == [
        {"observation": "000001", "text": "Réglages"},
        {"observation": "000003"},
    ]


@pytest.mark.parametrize(
    "torn", ['{"observation": "000003", "ide', '{"observation": "000003"}'], ids=["cut", "whole"]
)
def test_a_last_line_without_its_newline_is_left_out_and_cut_off_by_the_next_append(
    torn: str, tmp_path: Path
) -> None:
    path = tmp_path / "demo_identify.jsonl"
    path.write_text('{"observation": "000001"}\n' + torn, encoding="utf-8")

    assert graph_files.read_json_lines(path) == [{"observation": "000001"}]

    graph_files.append_json_line(path, {"observation": "000005"})

    assert graph_files.read_json_lines(path) == [
        {"observation": "000001"},
        {"observation": "000005"},
    ]


def test_an_append_after_a_torn_first_line_starts_the_file_again(tmp_path: Path) -> None:
    path = tmp_path / "demo_identify.jsonl"
    path.write_text('{"observ', encoding="utf-8")

    graph_files.append_json_line(path, {"observation": "000005"})

    assert path.read_text(encoding="utf-8") == '{"observation": "000005"}\n'


@pytest.mark.parametrize("line", ['{"observ', "[1, 2]", ""], ids=["not json", "a list", "empty"])
def test_a_damaged_line_before_the_last_raises_naming_the_file_and_the_line(
    line: str, tmp_path: Path
) -> None:
    path = tmp_path / "demo_identify.jsonl"
    path.write_text(f'{{"observation": "000001"}}\n{line}\n{{}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2 is not a JSON object") as raised:
        graph_files.read_json_lines(path)

    assert str(raised.value).startswith(str(path))


TEXT_MODEL = "provider/embedding"


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(None, "is missing", id="missing"),
        pytest.param("{not json", "cannot be read or parsed", id="corrupt"),
        pytest.param(
            json.dumps({"embeddings": {"000000": [0.5]}}),
            "is not tagged with the model that wrote it",
            id="no tag",
        ),
        pytest.param(
            json.dumps({"model": TEXT_MODEL}), "has no embeddings object", id="no embeddings"
        ),
        pytest.param(
            json.dumps({"model": "other/model", "embeddings": {"000000": [0.5]}}),
            f"was written by model 'other/model', not the configured {TEXT_MODEL!r}",
            id="another model",
        ),
        pytest.param(
            json.dumps({"model": TEXT_MODEL, "embeddings": {"000000": [0.5], "000001": []}}),
            "has a malformed or non-finite vector for observation(s) 000001",
            id="malformed vector",
        ),
        pytest.param(
            json.dumps({"model": TEXT_MODEL, "embeddings": {"000001": [0.5]}}),
            "has no vector for observation(s) 000000",
            id="a required vector missing",
        ),
    ],
)
def test_an_unusable_text_vector_file_raises_naming_the_file_and_the_problem(
    content: str | None, problem: str, tmp_path: Path
) -> None:
    """Nothing embeds a description again, so the graph cannot be loaded."""
    path = graph_files.text_embeddings_path(tmp_path / "demo.json")
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"demo_text_emb\.json") as raised:
        graph_files.load_embeddings(path, model=TEXT_MODEL, kind="text", required=["000000"])

    assert f"Text embedding file {path} {problem}" in str(raised.value)
    assert str(raised.value).endswith(". Explore the app again into a new graph directory.")


def test_a_step_keeps_its_record_and_image_under_its_id(tmp_path: Path) -> None:
    assert graph_files.step_record_path(tmp_path, "000007") == tmp_path / "000007.json"
    assert graph_files.observation_image_path(tmp_path, "000007") == tmp_path / "000007.webp"


def test_observation_image_b64_reads_and_encodes_the_image(tmp_path: Path) -> None:
    (tmp_path / "000003.webp").write_bytes(b"webp-bytes")
    assert graph_files.observation_image_b64(tmp_path, "000003") == base64.b64encode(
        b"webp-bytes"
    ).decode("ascii")


def test_observation_image_b64_raises_for_a_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"000003\.webp"):
        graph_files.observation_image_b64(tmp_path, "000003")


def test_require_known_edge_endpoints_accepts_a_graph_without_edges() -> None:
    assert graph_files.require_known_edge_endpoints(
        {"nodes": [{"id": "n1"}], "edges": []}, Path()
    ) == {"nodes": [{"id": "n1"}], "edges": []}


def test_require_known_edge_endpoints_raises_for_an_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="ghost"):
        graph_files.require_known_edge_endpoints(
            {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "ghost"}]},
            Path("g.json"),
        )


def test_require_known_edge_endpoints_raises_for_a_node_without_an_id(tmp_path: Path) -> None:
    """require_graph_shape runs first, so a node missing "id" entirely is
    reported here too, with the path, rather than silently skipped and left
    for a loader that runs later to report as a bare KeyError.
    """
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match="node id must be a string") as excinfo:
        graph_files.require_known_edge_endpoints({"nodes": [{"activity": "x"}], "edges": []}, path)
    assert str(path) in str(excinfo.value)


def test_require_graph_shape_accepts_string_ids_and_endpoints() -> None:
    assert graph_files.require_graph_shape(
        {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n1"}]}, Path()
    ) == {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n1"}]}


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"edges": []}, "must contain list fields 'nodes' and 'edges'"),
        ({"nodes": []}, "must contain list fields 'nodes' and 'edges'"),
        ({}, "must contain list fields 'nodes' and 'edges'"),
        ({"nodes": None, "edges": []}, "must contain list fields 'nodes' and 'edges'"),
        ({"nodes": [], "edges": None}, "must contain list fields 'nodes' and 'edges'"),
        ({"nodes": {}, "edges": []}, "must contain list fields 'nodes' and 'edges'"),
        ({"nodes": ["s0"], "edges": []}, "node must be an object"),
        ({"nodes": [], "edges": ["e0"]}, "edge must be an object"),
    ],
)
def test_require_graph_shape_rejects_a_malformed_shape(
    tmp_path: Path, data: dict[str, Any], match: str
) -> None:
    """Every shape failure, an absent ``"nodes"``, ``"nodes": null`` and
    ``"nodes": ["s0"]`` included, raises the ``ValueError`` every load failure raises,
    naming the path.
    """
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match=match) as excinfo:
        graph_files.require_graph_shape(data, path)
    assert str(path) in str(excinfo.value)
    assert "Explore the app again into a new graph directory" in str(excinfo.value)


def test_require_graph_shape_rejects_a_non_string_node_id(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match="node id must be a string") as excinfo:
        graph_files.require_graph_shape({"nodes": [{"id": 5}], "edges": []}, path)
    assert str(path) in str(excinfo.value)


def test_require_graph_shape_rejects_a_node_without_an_id(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match="node id must be a string") as excinfo:
        graph_files.require_graph_shape({"nodes": [{"activity": "x"}], "edges": []}, path)
    assert str(path) in str(excinfo.value)


def test_require_graph_shape_rejects_an_empty_node_id(tmp_path: Path) -> None:
    """An empty id passes the string check, so it is rejected on its own,
    naming the path, instead of loading as a node.
    """
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match="node id must be a non-empty string") as excinfo:
        graph_files.require_graph_shape({"nodes": [{"id": ""}], "edges": []}, path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    "edge",
    [{"source": None, "target": "n2"}, {"source": "n1", "target": None}],
)
def test_require_graph_shape_rejects_a_non_string_edge_endpoint(
    tmp_path: Path, edge: dict[str, Any]
) -> None:
    path = tmp_path / "demo.json"
    with pytest.raises(ValueError, match="edge endpoint must be a string") as excinfo:
        graph_files.require_graph_shape({"nodes": [], "edges": [edge]}, path)
    assert str(path) in str(excinfo.value)


def test_require_graph_shape_rejects_a_non_object_document(tmp_path: Path) -> None:
    """A file whose top level is a list, null or a string fails here with the
    path, not with an AttributeError out of ``.get``.
    """
    path = tmp_path / "graph.json"
    with pytest.raises(ValueError, match="must be an object") as excinfo:
        graph_files.require_graph_shape([], path)
    assert str(path) in str(excinfo.value)

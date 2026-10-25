"""Observed arrivals qualify the preparation retained for recorded actions."""

from __future__ import annotations

from android_app_graph.utils.graph_manager import EdgeDetails, GraphManager, SetupTrail


def _graph() -> GraphManager:
    graph = GraphManager()
    for node in ("s0", "s1"):
        graph.graph.add_node(node, replay_misses=0, observations=[f"{node}-entry"])
    return graph


def test_preparation_retains_its_observed_arrival_through_a_round_trip() -> None:
    trail = SetupTrail(lambda node: node)
    enter = {"action": "tap", "x": 20, "y": 30}
    trail.begin("s0", {"folder": "root"}, observation="root-entry")
    trail.step([enter], "s0", {"folder": "Pictures"}, observation="pictures")
    trail.step([{"action": "tap", "x": 20, "y": 50}], "s1", {}, observation="viewer")
    trail.step([{"action": "back"}], "s0", {"folder": "Pictures"}, observation="returned")
    assert trail.setup("s0") == [enter]
    assert trail.setup_observation("s0") == "root-entry"


def test_unknown_arrival_does_not_gain_an_anchor_from_its_capture() -> None:
    trail = SetupTrail(lambda node: node)
    trail.begin("s0", None, observation="unknown")
    trail.step([{"action": "tap", "x": 20, "y": 30}], "s0", {}, observation="later")
    assert trail.setup("s0") is None
    assert trail.setup_observation("s0") is None


def test_shorter_preparation_from_another_arrival_does_not_replace_the_anchor() -> None:
    graph = _graph()
    action = {"action": "tap", "x": 20, "y": 50}
    prefix = [{"action": "tap", "x": 20, "y": 30}]
    graph.add_edge("s0", "s1", action, details=EdgeDetails(setup=prefix, setup_observation="root"))
    graph.add_edge("s0", "s1", action, details=EdgeDetails(setup=[], setup_observation="Pictures"))
    edge = graph.graph["s0"]["s1"]
    assert edge["actions"] == [action]
    assert edge["setups"] == [prefix]
    assert edge["setup_observations"] == ["root"]
    path = graph.find_path("s0", "s1")
    assert path is not None
    assert path[1].setup_observation == "Pictures"


def test_reconstructed_menu_preparation_keeps_its_openers_anchor() -> None:
    graph = _graph()
    opener = {"action": "tap", "x": 90, "y": 10}
    graph.add_edge(
        "s0",
        "s0",
        opener,
        details=EdgeDetails(
            setup=[],
            setup_observation="closed-menu",
            schema_delta={"open_menu": {"before": None, "after": "overflow"}},
        ),
    )
    graph.add_edge(
        "s0",
        "s1",
        {"action": "tap", "x": 80, "y": 40},
        details=EdgeDetails(open_menu="overflow"),
    )
    path = graph.find_path("s0", "s1")
    assert path is not None
    assert path[1].setup == (opener,)
    assert path[1].setup_observation == "closed-menu"


def test_same_arrival_can_shorten_preparation_and_unknown_can_gain_one() -> None:
    graph = _graph()
    action = {"action": "tap", "x": 20, "y": 50}
    graph.add_edge("s0", "s1", action)
    graph.add_edge(
        "s0",
        "s1",
        action,
        details=EdgeDetails(setup=[{"action": "back"}], setup_observation="root"),
    )
    assert graph.graph["s0"]["s1"]["setup_observations"] == ["root"]
    graph.add_edge("s0", "s1", action, details=EdgeDetails(setup=[], setup_observation="root"))
    assert graph.graph["s0"]["s1"]["setups"] == [[]]
    assert graph.graph["s0"]["s1"]["setup_observations"] == ["root"]


def test_overlong_preparation_drops_its_anchor_with_its_inputs() -> None:
    graph = _graph()
    graph.add_edge(
        "s0",
        "s1",
        {"action": "tap"},
        details=EdgeDetails(setup=[{"action": "back"}] * 9, setup_observation="root"),
    )
    assert graph.graph["s0"]["s1"]["setups"] == [None]
    assert graph.graph["s0"]["s1"]["setup_observations"] == [None]


def test_a_shorter_known_arrival_remains_available_without_erasing_the_first_preparation() -> None:
    graph = _graph()
    action = {"action": "tap", "x": 20, "y": 50}
    prefix = [{"action": "tap", "x": 20, "y": 30}]
    graph.add_edge("s0", "s1", action, details=EdgeDetails(setup=prefix, setup_observation="root"))
    graph.add_edge("s0", "s1", action, details=EdgeDetails(setup=[], setup_observation="Pictures"))
    path = graph.find_path("s0", "s1")
    assert path is not None
    assert path[1].setup == ()
    assert path[1].setup_observation == "Pictures"
    assert graph.graph["s0"]["s1"]["setups"] == [prefix]
    assert graph.graph["s0"]["s1"]["setup_observations"] == ["root"]


def test_preparation_choices_are_bounded_by_strictly_decreasing_setup_length() -> None:
    graph = _graph()
    action = {"action": "tap"}
    for length in range(8, -1, -1):
        graph.add_edge(
            "s0",
            "s1",
            action,
            details=EdgeDetails(
                setup=[{"action": "back"}] * length, setup_observation=f"entry-{length}"
            ),
        )
    edge = graph.graph["s0"]["s1"]
    assert len(edge["actions"]) == 1
    assert len(edge["setup_alternatives"][0]) == 8
    for length in range(9):
        graph.add_edge(
            "s0",
            "s1",
            action,
            details=EdgeDetails(
                setup=[{"action": "back"}] * length, setup_observation=f"another-{length}"
            ),
        )
    assert len(edge["setup_alternatives"][0]) == 8
    graph.add_edge("s0", "s1", action)
    assert len(edge["setup_alternatives"][0]) == 8


def test_an_existing_alternative_can_shorten_without_erasing_other_entries() -> None:
    graph = _graph()
    action = {"action": "tap"}
    for anchor, length in [("root", 4), ("folder", 3), ("folder", 2), ("folder", 3), ("root", 1)]:
        graph.add_edge(
            "s0",
            "s1",
            action,
            details=EdgeDetails(setup=[{"action": "back"}] * length, setup_observation=anchor),
        )
    edge = graph.graph["s0"]["s1"]
    assert len(edge["setups"][0]) == 1
    assert edge["setup_alternatives"] == [
        [{"setup": [{"action": "back"}] * 2, "observation": "folder"}]
    ]


def test_menu_reconstruction_retains_each_entry_of_the_chosen_opener() -> None:
    graph = _graph()
    opener = {"action": "tap", "x": 90, "y": 10}
    delta = {"open_menu": {"before": None, "after": "overflow"}}
    graph.add_edge(
        "s0",
        "s0",
        opener,
        details=EdgeDetails(
            setup=[{"action": "back"}], setup_observation="root", schema_delta=delta
        ),
    )
    graph.add_edge(
        "s0",
        "s0",
        opener,
        details=EdgeDetails(setup=[], setup_observation="folder", schema_delta=delta),
    )
    graph.add_edge(
        "s0", "s1", {"action": "tap", "x": 80, "y": 40}, details=EdgeDetails(open_menu="overflow")
    )
    step = graph.replay_step("s0", "s1", 0)
    assert step is not None
    assert step.setup == (opener,)
    assert step.setup_observation == "folder"

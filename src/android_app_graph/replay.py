"""Replaying a recorded action, as exploration and the audit do to reach a node.

A replayed tap or long press goes to the element it acted on when it was recorded:
the screen is read first, and the action goes to the centre of that element's current
bounds when its anchor finds the element there, as ``resolve_anchor`` decides.
Otherwise it goes to the recorded coordinates, and the log says why. Typing, swipes,
Back, Home, Enter and waits replay as recorded.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from android_app_graph.step_log import Observation
from android_app_graph.ui_tree import Anchor, AnchorMiss, Bounds, resolve_anchor

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    from android_app_graph.step_log import StepRecorder

_ANCHORED = frozenset({"tap", "long_press"})
_SUB_ACTION_PAUSE_SECONDS = 0.5
_SETTLE_SECONDS = 1


def replay_action(
    recorder: StepRecorder,
    action: dict[str, Any] | list[dict[str, Any]],
    anchors: Sequence[Anchor | None],
    *,
    logger: logging.Logger,
    screen: Observation | None = None,
) -> list[dict[str, Any]]:
    """Send ``action``, an edge's recorded action, whose sub-actions have ``anchors``,
    then wait a second for the screen to settle, and return the sub-actions as sent.

    Before a tap or long press with an anchor, the screen is read through ``recorder``,
    so the step log keeps it under the recorder's source, and the action goes to the
    centre of the anchor's element in that screen's tree, keeping its other fields,
    such as a long press's duration. Without an anchor, a tree or a unique element a
    tap at that centre reaches, it goes to the recorded coordinates. Each tap and long
    press logs which way it went. ``screen`` is the observation of the screen the
    action starts on when the caller has just taken it; the first sub-action uses it
    instead of reading the screen again. The sub-actions of a compound action are half
    a second apart, and ``{}``, an action an edge was saved without, sends nothing.
    """
    sub_actions = action if isinstance(action, list) else [action] if action else []
    sent: list[dict[str, Any]] = []
    current = screen
    for position, sub_action in enumerate(sub_actions):
        anchor = anchors[position] if position < len(anchors) else None
        resolved = _resolved(recorder, sub_action, anchor, current, logger)
        recorder.exe_action(resolved)
        sent.append(resolved)
        current = None
        if isinstance(action, list):
            time.sleep(_SUB_ACTION_PAUSE_SECONDS)
    time.sleep(_SETTLE_SECONDS)
    return sent


def _resolved(
    recorder: StepRecorder,
    action: dict[str, Any],
    anchor: Anchor | None,
    screen: Observation | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Return ``action`` as a replay sends it: a tap or long press at its anchor's
    current centre when the screen shows that element, else as recorded."""
    kind = action.get("action")
    if kind not in _ANCHORED:
        return action
    recorded = (action.get("x"), action.get("y"))
    if anchor is None:
        logger.info("    %s at %s: recorded coordinates: no_anchor", kind, recorded)
        return action
    found = _anchor_point(screen or Observation.from_state(recorder.get_state()), anchor)
    if isinstance(found, str):
        logger.info(
            '    %s at %s: recorded coordinates: %s "%s"',
            kind,
            recorded,
            found,
            anchor.locator.label,
        )
        return action
    x, y = found
    logger.info(
        '    %s at %s: anchor "%s", recorded at %s', kind, found, anchor.locator.label, recorded
    )
    return {**action, "x": x, "y": y}


def _anchor_point(
    observation: Observation, anchor: Anchor
) -> tuple[int, int] | AnchorMiss | Literal["no_tree"]:
    tree = observation.ui_tree
    if tree is None:
        return "no_tree"
    width, height = observation.screen_size
    return resolve_anchor(tree, anchor, observation.package, Bounds(0, 0, width, height))

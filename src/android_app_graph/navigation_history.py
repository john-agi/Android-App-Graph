"""Bounded factual context for repeated ordinary exploration routes."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteState:
    """A graph identity and the recorded pixels below the status bar."""

    node: str
    pixels: str


@dataclass(frozen=True, slots=True)
class _Transition:
    before: RouteState
    after: RouteState
    action: str


class NavigationHistory:
    """Remember eight contiguous known-edge transitions, without inferring exhaustion."""

    def __init__(self, resolve_node: Callable[[str], str]) -> None:
        self._resolve_node = resolve_node
        self._transitions: deque[_Transition] = deque(maxlen=8)

    def clear(self) -> None:
        """Start fresh when ordinary observed navigation is interrupted."""
        self._transitions.clear()

    def record(
        self,
        before: RouteState,
        after: RouteState,
        action: str,
        *,
        known_edge: bool,
    ) -> None:
        """Retain a known transition whose observations and identities remain available."""
        self._validate(before)
        if not known_edge or not before.pixels or not after.pixels:
            self.clear()
            return
        self._transitions.append(_Transition(before, after, action))

    def _validate(self, current: RouteState) -> None:
        if self._transitions and (
            self._transitions[-1].after != current
            or any(
                self._resolve_node(state.node) != state.node
                for transition in self._transitions
                for state in (transition.before, transition.after)
            )
        ):
            # Rewriting old identities after a merge could manufacture a repeated route.
            self.clear()

    def feedback(self, current: RouteState) -> str:
        """Describe two identical closed routes of two through four ordinary steps."""
        self._validate(current)
        transitions = list(self._transitions)
        for length in range(2, 5):
            if len(transitions) < 2 * length:
                continue
            previous, recent = transitions[-2 * length : -length], transitions[-length:]
            states = [transition.before for transition in recent]
            if (
                states != [transition.before for transition in previous]
                or recent[-1].after != states[0]
                or len({state.node for state in states}) == 1
            ):
                continue
            route = recent[0].before.node
            for transition in recent:
                family = (
                    transition.action
                    if transition.action
                    in {"tap", "back", "swipe", "long_press", "type", "enter", "menu"}
                    else "other"
                )
                route += f" --{family}--> {transition.after.node}"
            return (
                f"Recent repeated route (last {2 * length} ordinary steps; same recorded pixels "
                f"at corresponding visits; latest cycle actions): {route}. Hidden state may still "
                "change; repetition "
                "does not establish exhaustion. Continue a useful pending workflow when needed. "
                "Otherwise consider a relevant untried local control within the existing action "
                "policy. Preserve pending edits and apply the same mutation rules.\n"
            )
        return ""

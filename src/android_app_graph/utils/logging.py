"""Rich-based colored logging for Android-App-Graph.

Each subsystem gets a consistent visual identity so the exploration trace is
easy to scan in real time:
  - Explorer            → cyan
  - Kobe                → magenta
  - Graph               → green
  - VLM                 → yellow
  - Default             → slate
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import override

from rich.console import Console
from rich.text import Text

# Logger name prefix -> style config.
_LOGGER_CONFIG: dict[str, dict[str, str]] = {
    "scripts": {
        "label": "Explorer",
        "badge_style": "bold white on cyan",
        "text_style": "cyan",
    },
    "__main__": {
        "label": "Explorer",
        "badge_style": "bold white on cyan",
        "text_style": "cyan",
    },
    "android_app_graph.explorer": {
        "label": "Explorer",
        "badge_style": "bold white on cyan",
        "text_style": "cyan",
    },
    "android_app_graph.kobe": {
        "label": "Kobe",
        "badge_style": "bold white on magenta",
        "text_style": "magenta",
    },
    "android_app_graph.utils.graph_manager": {
        "label": "Graph",
        "badge_style": "bold white on green",
        "text_style": "green",
    },
    "android_app_graph.utils.vlm_utils": {
        "label": "VLM",
        "badge_style": "bold black on yellow",
        "text_style": "yellow",
    },
}

_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: "bold white on bright_black",
    logging.INFO: "bold white on blue",
    logging.WARNING: "bold black on yellow",
    logging.ERROR: "bold white on red",
    logging.CRITICAL: "bold white on dark_red",
}

_console = Console(stderr=True, force_terminal=True)


class _ColorHandler(logging.Handler):
    """Custom handler that renders compact, color-coded Rich log lines."""

    default_time_style = "grey62"
    default_text_style = "white"
    dim_detail_style = "grey50"

    def _resolve_component(self, record: logging.LogRecord) -> dict[str, str]:
        component = {
            "label": record.name.split(".")[-1].replace("_", "-").title(),
            "badge_style": "bold white on grey23",
            "text_style": self.default_text_style,
        }
        for prefix, cfg in _LOGGER_CONFIG.items():
            if record.name == prefix or record.name.startswith(prefix + "."):
                component.update(cfg)
                break
        return component

    def _append_message(self, line: Text, message: str, text_style: str) -> None:
        parts = message.splitlines() or [message]
        line.append(parts[0], style=text_style)
        for extra_line in parts[1:]:
            line.append("\n")
            line.append(" " * 24, style=self.dim_detail_style)
            line.append(extra_line, style=text_style)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            component = self._resolve_component(record)
            level_style = _LEVEL_STYLES.get(record.levelno, "white")

            line = Text()

            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            line.append(f"{ts} ", style=self.default_time_style)

            level_label = record.levelname[:4].ljust(4)
            line.append(f" {level_label} ", style=level_style)
            line.append(" ", style=self.default_time_style)

            line.append(
                f" {component['label']:^10} ",
                style=component["badge_style"],
            )
            line.append(" ", style=self.default_time_style)

            msg = record.getMessage()
            self._append_message(line, msg, component["text_style"])

            if record.exc_info and not record.exc_text:
                record.exc_text = "".join(traceback.format_exception(*record.exc_info))
            if record.exc_text:
                line.append("\n")
                line.append(record.exc_text.rstrip(), style="bold red")

            _console.print(line, highlight=False)
        except Exception:  # noqa: BLE001  # required by the logging.Handler.handleError contract
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with Rich colored output.

    Call this once at the entry point (explore.py / cli.py).
    """
    handler = _ColorHandler()

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy HTTP loggers from OpenAI SDK / httpx / urllib3
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

"""Required model configuration fails before device work or provider requests."""

from __future__ import annotations

import pytest

from android_app_graph.commands import audit, embed, plot


@pytest.mark.parametrize("command", ["audit", "embed", "plot"])
def test_command_help_needs_no_model_configuration(command: str) -> None:
    main = {"audit": audit.main, "embed": embed.main, "plot": plot.main}[command]
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0

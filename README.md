# Android-App-Graph

Android-App-Graph is a graph-based exploration and runtime framework for mobile
GUI agents, forked from [UI-KOBE](https://github.com/YuxiangChai/UI-KOBE) by
Yuxiang Chai. It explores Android apps, builds state-transition graphs, and lets
downstream agents use those graphs to navigate apps more reliably.

The repository has two main uses:

1. Build a UI knowledge graph for an Android app.
2. Use the graph inside [AITK](https://github.com/YuxiangChai/AITK) or
   [Android World](https://github.com/google-research/android_world) agents.

## How It Works

1. **Explore an app.** UI-KOBE opens an Android app on an emulator, observes the
   current screen, chooses an exploration action, and records the result.
2. **Build a graph.** Each app state becomes a graph node. Actions that move the
   app between states become graph edges.
3. **Run graph-guided agents.** The UI-KOBE v2 runtime loads the graph, matches
   the current screen to a node, and chooses the next action from nearby graph
   edges or a free-form fallback action.

## Repository Layout

```text
UI-KOBE/
├── aitk_files/
│   └── ui_kobe_v2.py              # AITK translator to copy into AITK
├── aw_files/
│   └── ui_kobe_aw_agent.py        # Android World agent adapter
├── configs/
│   └── explore.yaml               # Sanitized exploration demo config
├── scripts/
│   ├── audit_graph.py             # Optional graph audit utility
│   ├── explore.py                 # Script entry point for exploration
│   ├── plot_graph.py              # Graph visualization utility
│   ├── precompute_graph_image_embeddings.py
│   └── run_explore.sh             # Auto-resume wrapper
└── ui_kobe/
    ├── cli.py                     # kobe-explore command
    ├── kobe.py                    # Core app explorer
    └── utils/                     # Graph, VLM, and logging helpers
```

`ui_kobe/` is the installable Python package used by the exploration CLI and by
the AITK translator. Keep it installed in the same environment as AITK.

Generated graphs, logs, and outputs are intentionally ignored by git.

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) for the UI-KOBE development environment
- [AITK](https://github.com/YuxiangChai/AITK), installed before UI-KOBE
- Android SDK with `adb` and `emulator` on `PATH`
- An Android Virtual Device with the target app installed
- VLM provider credentials exported as environment variables

## Install

UI-KOBE depends on AITK. Install AITK first, then install UI-KOBE into the same
Python environment.

### Option 1: uv with sibling repositories

```bash
git clone https://github.com/YuxiangChai/AITK.git
git clone https://github.com/YuxiangChai/UI-KOBE.git

cd UI-KOBE
uv sync
```

The default `uv` setup expects `AITK` and `UI-KOBE` to be side by side because
`pyproject.toml` points to `../AITK` as the editable AITK source.

Run commands through `uv`:

```bash
uv run kobe-explore --help
```

### Option 2: Existing AITK conda environment

If AITK runs from a conda environment, install UI-KOBE into that same
environment:

```bash
conda activate <aitk-env>

cd /path/to/AITK
pip install -e .

cd /path/to/UI-KOBE
pip install -e .
```

This makes `ui_kobe` importable to AITK translators copied into
`AITK/aitk/translators/`.

## Configure API Credentials

Do not put API keys directly in config files. Use a local `.env` file instead:

```bash
cp .env.example .env
```

Edit `.env` and fill in your provider credentials. The template uses the
previous UI-KOBE demo defaults:

```bash
UI_KOBE_PAGE_DETAIL_MODEL=gpt-5.4
UI_KOBE_EMBEDDING_MODEL=gemini-embedding-2-preview
UI_KOBE_INSTRUCTION_MODEL=gpt-5.4
UI_KOBE_ACTION_MODEL=qwen3.5-plus-2026-02-15
```

Load the file before running commands:

```bash
set -a
source .env
set +a
```

`.env` is ignored by git. You can use any OpenAI-compatible endpoint for the
chat and text embedding providers; update model names and base URLs in `.env`
to match your provider.

## Explore an App

Edit `configs/explore.yaml` for your emulator and target app:

```yaml
device:
  udid: emulator-5554
  avd_name: AndroidWorldAvd

apps:
  - name: demo_app
    package_name: com.example.app
```

Run exploration:

```bash
set -a && source .env && set +a
uv run kobe-explore -c configs/explore.yaml
```

Useful options:

```bash
uv run kobe-explore -c configs/explore.yaml --max-steps 100
uv run kobe-explore -c configs/explore.yaml --resume-from auto --max-steps 50
```

Exploration writes graph files under `graphs/<app_name>/`.

For long runs, use the auto-resume wrapper:

```bash
./scripts/run_explore.sh -c configs/explore.yaml
```

## Visualize or Audit a Graph

Create an HTML graph visualization:

```bash
uv run python scripts/plot_graph.py graphs/<app_name>/<app_name>.json
```

Run the graph audit utility:

```bash
uv run python scripts/audit_graph.py -c configs/explore.yaml --app <app_name>
```

Precompute native Gemini image embeddings for a graph:

```bash
uv run python scripts/precompute_graph_image_embeddings.py \
  --config configs/explore.yaml \
  --graph graphs/<app_name>/<app_name>.json
```

## Use UI-KOBE with AITK

1. Install AITK and UI-KOBE in the same Python environment.
2. Copy the v2 translator into AITK:

```bash
cp /path/to/UI-KOBE/aitk_files/ui_kobe_v2.py \
  /path/to/AITK/aitk/translators/ui_kobe_v2.py
```

3. Configure your AITK agent to use `ui_kobe_v2` and pass the graph directory
   plus VLM settings through `translator_args`.

Example translator args:

```yaml
translator_args:
  graph_dir: /path/to/UI-KOBE/graphs
  vlm_config:
    similarity_threshold: 0.84
    page_detail:
      model: ${UI_KOBE_PAGE_DETAIL_MODEL}
      base_url: ${UI_KOBE_PAGE_DETAIL_BASE_URL}
      api_key: ${UI_KOBE_PAGE_DETAIL_API_KEY}
    embedding:
      model: ${UI_KOBE_EMBEDDING_MODEL}
      base_url: ${UI_KOBE_EMBEDDING_BASE_URL}
      api_key: ${UI_KOBE_EMBEDDING_API_KEY}
    instruction:
      model: ${UI_KOBE_INSTRUCTION_MODEL}
      base_url: ${UI_KOBE_INSTRUCTION_BASE_URL}
      api_key: ${UI_KOBE_INSTRUCTION_API_KEY}
    action:
      model: ${UI_KOBE_ACTION_MODEL}
      base_url: ${UI_KOBE_ACTION_BASE_URL}
      api_key: ${UI_KOBE_ACTION_API_KEY}
    image_embedding:
      model: models/gemini-embedding-exp-03-07
      native_base_url: https://generativelanguage.googleapis.com/v1beta
      api_key: ${GEMINI_API_KEY}
```

The copied translator imports helper functions from the installed `ui_kobe`
package, so UI-KOBE must be installed in the AITK environment.

## Use UI-KOBE with Android World

1. Install Android World following its upstream instructions.
2. Install AITK and UI-KOBE in the same environment used by Android World.
3. Copy the Android World adapter:

```bash
cp /path/to/UI-KOBE/aw_files/ui_kobe_aw_agent.py \
  /path/to/android_world/android_world/agents/ui_kobe_aw_agent.py
```

4. Register the agent in Android World's `run.py`.

Add the import:

```python
from android_world.agents import ui_kobe_aw_agent
```

Add a branch inside `_get_agent`:

```python
elif _AGENT_NAME.value == 'ui_kobe':
  agent = ui_kobe_aw_agent.UIKobeAndroidWorldAgent.from_config(
      env,
      '/path/to/UI-KOBE/configs/explore.yaml',
  )
```

5. Run Android World with the UI-KOBE agent:

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=ui_kobe \
  --perform_emulator_setup \
  --tasks=ContactsAddContact
```

The Android World adapter reuses the AITK translator runtime and converts AITK
actions into Android World `JSONAction` objects.

## Create Custom Agents

The included AITK translator and Android World adapter are reference
implementations for using a UI-KOBE graph inside an interactive agent loop. You
do not need to copy their framework choices exactly. For a custom agent, reuse
the graph workflow shown in these files and adapt it to your own runtime,
benchmark, simulator, browser tool, desktop controller, or mobile automation
system.

The core pattern is:

1. Load a UI-KOBE graph for the target app.
2. Match the current observation to a graph node.
3. Read nearby edges and state metadata as action affordances.
4. Choose either a graph-guided transition or a free-form fallback action.
5. Convert that decision into the action format required by your environment.

The two files below show this pattern in concrete systems.

### AITK translator

Create a translator in `AITK/aitk/translators/` that implements the AITK
translator interface:

```python
from aitk.translators.base import BaseTranslator


class MyTranslator(BaseTranslator):
    def to_agent(self, task: str, state: dict, history: dict) -> str:
        return '{"message": "Tap search", "aitk_action": {"action": "tap", "x": 100, "y": 200}}'

    def to_device(self, action: str, width: int, height: int) -> dict:
        return {"action": "wait", "time": 1}


def register(kargs: dict) -> MyTranslator:
    return MyTranslator(**kargs)
```

Use `aitk_files/ui_kobe_v2.py` as the full graph-guided AITK example. It shows
how to load a UI-KOBE graph, identify the current node, record task-relevant
information, choose a graph edge, and convert the choice into a low-level AITK
action. You can transfer the same graph usage to other agent frameworks by
replacing only the final action-conversion layer.

### Android World agent

Create an Android World agent by subclassing `EnvironmentInteractingAgent` and
implementing `step(goal)`. The step should:

1. Read the current state with `get_post_transition_state()`.
2. Choose one action for the current goal.
3. Execute the action with `self.env.execute_action(...)`.
4. Return `AgentInteractionResult(done=<bool>, data=<dict>)`.

Use `aw_files/ui_kobe_aw_agent.py` as an example of adapting the same graph
runtime to another interactive system. It wraps the UI-KOBE AITK translator and
maps its output into Android World's `JSONAction` format. For other tools, write
an equivalent adapter that maps graph-guided decisions into that tool's action
space.

## Acknowledgements

Android-App-Graph is a fork of [UI-KOBE](https://github.com/YuxiangChai/UI-KOBE)
by [Yuxiang Chai](https://github.com/YuxiangChai), released under the
[Apache License 2.0](LICENSE). Files from the original work have been modified
in this repository; the attribution notice is in [NOTICE](NOTICE).
[AITK](https://github.com/YuxiangChai/AITK), the agent framework this project
plugs into, is by the same author.

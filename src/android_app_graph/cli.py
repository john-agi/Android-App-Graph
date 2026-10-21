import argparse
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from android_app_graph.device import (
    AdbController,
    AvdManager,
    EmulatorManager,
    grant_runtime_permissions,
    launch_app,
)
from android_app_graph.kobe import EXPLORATION_ROLES, ExplorerConfig, Kobe
from android_app_graph.utils import (
    check_config_keys,
    open_client,
    resolve_api_key,
    resolve_model_settings,
)
from android_app_graph.utils.logging import setup_logging


def check_avd(avd_manager: AvdManager, config: dict[str, Any], logger: logging.Logger) -> None:
    """Ensure an AVD is running. If not, duplicate the base AVD and launch it."""
    running_avd_list = avd_manager.get_running_avd_list()
    if running_avd_list:
        logger.info("Running AVD found.")
        return

    avd_name = config["device"]["avd_name"]
    dup_name = f"{avd_name}_dup"

    logger.info("No running AVD found. Starting a new AVD duplicate...")
    avd_manager.duplicate_avd(avd_name)
    emulator = shutil.which("emulator")
    if emulator is None:
        msg = "emulator was not found on PATH"
        raise FileNotFoundError(msg)
    cmd = [
        emulator,
        "-avd",
        dup_name,
        "-no-snapshot",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603  # argv list without a shell; the AVD name comes from the operator's config
    logger.info("Waiting for AVD '%s' to start and load...", dup_name)
    time.sleep(60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore Android app UI with Android-App-Graph.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default="configs/explore.yaml",
        help="Path to the YAML config file (default: configs/explore.yaml).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max exploration steps from config.",
    )
    parser.add_argument(
        "--resume-from",
        "-r",
        type=str,
        default=None,
        help='Node ID to resume from, or "auto" for the top-ranked exploration target.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    vlm_config = config.get("vlm", {})
    try:
        check_config_keys(config)
        resolve_model_settings(vlm_config, roles=EXPLORATION_ROLES)
        api_key = resolve_api_key()
    except ValueError as exc:
        parser.error(str(exc))

    setup_logging(level=logging.INFO)
    logger = logging.getLogger("android_app_graph.explorer")

    check_avd(EmulatorManager(logger), config, logger)
    controller = AdbController(config, logger)

    exp_config = config.get("experiment", {})
    graph_dir = exp_config.get("graph_dir", "graphs")

    max_steps = args.max_steps or exp_config.get("max_steps", 20)
    resume_from = args.resume_from or exp_config.get("resume_from")
    coverage_checkpoint_steps = exp_config.get("coverage_checkpoint_steps", 50)
    coverage_checkpoint_top_k = exp_config.get("coverage_checkpoint_top_k", 15)

    with open_client(api_key) as client:
        for app in config["apps"]:
            app_name = app["name"]
            package_name = app["package_name"]

            logger.info("Terminating all running apps...")
            controller.terminate_all_apps()
            time.sleep(1)

            grant_runtime_permissions(config["device"]["udid"], package_name)
            launch_app(config, app, logger)
            time.sleep(4)

            explorer_config = ExplorerConfig(
                app_name=app_name,
                package_name=package_name,
                vlm_config=vlm_config,
                graph_dir=graph_dir,
                max_steps=max_steps,
                coverage_checkpoint_steps=coverage_checkpoint_steps,
                coverage_checkpoint_top_k=coverage_checkpoint_top_k,
                launch_activity=app.get("launch_activity"),
            )
            kobe = Kobe(controller, explorer_config, client=client, logger=logger)
            try:
                kobe.explore(resume_from=resume_from)
            finally:
                kobe.save_graph()


if __name__ == "__main__":
    main()

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from android_app_graph.device import (
    AdbController,
    AdbDisplayAccess,
    AdbServerUnreachableError,
    AvdManager,
    DeviceUnreachableError,
    EmulatorManager,
    StartupRecovery,
    grant_runtime_permissions,
    launch_app,
)
from android_app_graph.failure_monitor import AdbFailureReader, FailureMonitor
from android_app_graph.graph_files import app_graph_path
from android_app_graph.kobe import EXPLORATION_ROLES, ExplorerConfig, Kobe
from android_app_graph.payloads import as_object_dict
from android_app_graph.run_environment import RunBaselineError, RunEnvironment, mutation_config
from android_app_graph.utils import (
    check_config_keys,
    open_client,
    resolve_api_key,
    resolve_matching_settings,
    resolve_model_settings,
    resolve_unreachable_limit,
)
from android_app_graph.utils.logging import setup_logging


def check_avd(avd_manager: AvdManager, config: dict[str, Any], logger: logging.Logger) -> None:
    """Ensure an AVD is running. If not, duplicate the base AVD and launch it.

    An adb server that adb cannot reach raises ``AdbServerUnreachableError`` before
    anything is copied or started.
    """
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
        help='App-owned node ID to resume from, or "auto" for the top-ranked exploration target.',
    )
    return parser


def main() -> None:
    """Run app-graph. When adb cannot reach its server, or the device stays unreachable
    past ``device.unreachable_limit``, or a graph requires a matching baseline,
    exit with status 1 and a diagnostic."""
    try:
        _explore()
    except (AdbServerUnreachableError, DeviceUnreachableError, RunBaselineError) as exc:
        sys.exit(f"app-graph: error: {exc}")


def _explore() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    vlm_config = config.get("vlm", {})
    try:
        check_config_keys(config)
        mutation = mutation_config(as_object_dict(config.get("experiment", {})).get("mutation", {}))
        resolve_model_settings(vlm_config, roles=EXPLORATION_ROLES)
        resolve_matching_settings(vlm_config)
        unreachable_limit = resolve_unreachable_limit(config)
        api_key = resolve_api_key()
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    setup_logging(level=logging.INFO)
    logger = logging.getLogger("android_app_graph.explorer")

    check_avd(EmulatorManager(logger), config, logger)
    controller = AdbController(config, logger, unreachable_limit=unreachable_limit)
    recovery = StartupRecovery(controller, unreachable_limit, logger)

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

            graph_path = app_graph_path(Path(graph_dir), app_name)
            environment = RunEnvironment(
                controller,
                AdbDisplayAccess(config["device"]["udid"]),
                graph_path,
                mutation=mutation,
                logger=logger,
            )
            failure_monitor = FailureMonitor(
                AdbFailureReader(config["device"]["udid"], package_name),
                graph_path,
                package_name,
                config["device"]["udid"],
                logger,
            )
            with failure_monitor, environment as managed:
                logger.info("Terminating all running apps...")
                controller.terminate_all_apps(recovery)
                time.sleep(1)

                grant_runtime_permissions(config["device"]["udid"], package_name, recovery)
                launch_activity = launch_app(config, app, logger, recovery)
                time.sleep(4)

                explorer_config = ExplorerConfig(
                    app_name=app_name,
                    package_name=package_name,
                    vlm_config=vlm_config,
                    graph_dir=graph_dir,
                    max_steps=max_steps,
                    coverage_checkpoint_steps=coverage_checkpoint_steps,
                    coverage_checkpoint_top_k=coverage_checkpoint_top_k,
                    launch_activity=launch_activity,
                    unreachable_limit=unreachable_limit,
                )
                kobe = Kobe(managed, explorer_config, client=client, logger=logger)
                failure_monitor.step_run_id = kobe.controller.run_id
                try:
                    kobe.explore(resume_from=resume_from)
                finally:
                    kobe.save_graph()


if __name__ == "__main__":
    main()

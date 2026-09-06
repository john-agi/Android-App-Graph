import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from android_app_graph.device import AvdManager
from android_app_graph.kobe import Kobe
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
    cmd = [
        "emulator",
        "-avd",
        dup_name,
        "-no-snapshot",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Waiting for AVD '%s' to start and load...", dup_name)
    time.sleep(60)


def launch_app(config: dict[str, Any], app: dict[str, Any], logger: logging.Logger) -> None:
    """Launch an app using an explicit activity when configured."""
    udid = config["device"]["udid"]
    app_name = app["name"]
    package_name = app["package_name"]
    launch_activity = app.get("launch_activity")

    if launch_activity:
        logger.info("Launching app %s via activity %s ...", app_name, launch_activity)
        result = subprocess.run(
            [
                "adb",
                "-s",
                udid,
                "shell",
                "am",
                "start",
                "-n",
                launch_activity,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        logger.warning(
            "Activity launch failed for %s; falling back to package launch. error=%s",
            app_name,
            (result.stderr or result.stdout or "").strip(),
        )

    logger.info("Launching app %s (%s) ...", app_name, package_name)
    subprocess.run(
        [
            "adb",
            "-s",
            udid,
            "shell",
            "monkey",
            "-p",
            package_name,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


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
        help='Node ID to resume from, or "auto" for least-explored node.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from aitk.utils.adb_controller import ADBController
        from aitk.utils.avd_manager import AVDManager
    except ImportError:
        parser.exit(
            1,
            "app-graph requires the 'aitk' dependency. Install it first, then rerun the command.\n",
        )

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(level=logging.INFO)
    logger = logging.getLogger("android_app_graph.explorer")

    avd_manager = AVDManager(logger)
    check_avd(avd_manager, config, logger)
    controller = ADBController(config, logger)

    exp_config = config.get("experiment", {})
    vlm_config = config.get("vlm", {})
    graph_dir = exp_config.get("graph_dir", "graphs")

    max_steps = args.max_steps or exp_config.get("max_steps", 20)
    resume_from = args.resume_from or exp_config.get("resume_from")
    coverage_checkpoint_steps = exp_config.get("coverage_checkpoint_steps", 50)
    coverage_checkpoint_top_k = exp_config.get("coverage_checkpoint_top_k", 15)

    for app in config["apps"]:
        app_name = app["name"]
        package_name = app["package_name"]

        logger.info("Terminating all running apps...")
        controller._terminate_all_apps()
        time.sleep(1)

        launch_app(config, app, logger)
        time.sleep(4)

        kobe = Kobe(
            controller,
            app_name,
            package_name,
            logger,
            vlm_config=vlm_config,
            graph_dir=graph_dir,
            max_steps=max_steps,
            coverage_checkpoint_steps=coverage_checkpoint_steps,
            coverage_checkpoint_top_k=coverage_checkpoint_top_k,
        )
        try:
            kobe.explore(resume_from=resume_from)
        finally:
            kobe.save_graph()


if __name__ == "__main__":
    main()

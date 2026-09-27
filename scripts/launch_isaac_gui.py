from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISAAC_SIM = Path(
    os.environ.get("ISAAC_SIM", "/home/etfrobotics/isaacsim")
)

# Lock the repo-tested NumPy into sys.modules before Kit starts.
import numpy as np

EXPECTED_NUMPY = "1.26.4"
if np.__version__ != EXPECTED_NUMPY:
    raise RuntimeError(
        f"Expected NumPy {EXPECTED_NUMPY}, got {np.__version__} "
        f"from {np.__file__}. Reinstall .deps with requirements.txt."
    )

experience = ISAAC_SIM / "apps" / "isaacsim.exp.full.kit"
if not experience.is_file():
    raise FileNotFoundError(
        f"Isaac full experience not found: {experience}"
    )

print(
    "[robot_skill_stack.launcher] "
    f"NumPy {np.__version__} from {np.__file__}",
    flush=True,
)
print(
    "[robot_skill_stack.launcher] ROS 2 bridge disabled for this course runtime",
    flush=True,
)
print(
    "[robot_skill_stack.launcher] "
    f"Starting full Isaac experience: {experience}",
    flush=True,
)

from isaacsim import SimulationApp

extra_args = [
    # The full Isaac experience normally enables the ROS 2 bridge on Linux.
    # This project does not use ROS yet. Starting it from python.sh can pick up
    # an incompatible system Jazzy/rclpy install, so explicitly override the
    # full-app startup setting for this launcher.
    "--/isaac/startup/ros_bridge_extension=",
    "--/isaac/startup/ros_sim_control_extension=false",
    "--ext-folder",
    str(ROOT / "exts"),
    "--enable",
    "robot_skill_stack.ui",
    *sys.argv[1:],
]

simulation_app = SimulationApp(
    {
        "headless": False,
        "hide_ui": False,
        "extra_args": extra_args,
    },
    experience=str(experience),
)

try:
    while simulation_app.is_running():
        simulation_app.update()
finally:
    if not simulation_app.is_exiting():
        simulation_app.close()

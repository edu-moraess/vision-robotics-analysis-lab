import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation import RobotSimulation


def test_robot_simulation_moves_only_in_simulation_space():
    sim = RobotSimulation(step_px=5)
    state = sim.update(
        (100, 100), current_path=[(50, 90), (50, 50), (50, 10)],
        alternative_path=[(40, 90), (40, 10)], navigation_state="NAVIGATING",
        obstacles=[{"id": 1}], risk_zones=[{"zone_id": "GLOBAL"}],
    )
    assert state.status == "SIMULATION"
    assert state.navigation_state == "NAVIGATING"
    assert state.step_index == 1
    assert state.robot_position != (50.0, 88.0)
    assert "pixels" in " ".join(state.notes).lower()


def test_robot_simulation_overlay_is_marked():
    sim = RobotSimulation()
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    sim.update((80, 80), current_path=[(40, 70), (40, 10)], navigation_state="REPLANNING")
    overlay = sim.render(image)
    assert overlay.shape == image.shape
    assert int(overlay.sum()) > 0

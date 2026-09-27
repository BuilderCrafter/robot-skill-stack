from __future__ import annotations

import argparse
import sys

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.presentation import WorldModelViewModel
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.results import fail_result, write_result


def make_object(object_id, *, x, visible, source):
    return WorldObject(
        object_id=object_id,
        class_name="cube",
        pose=Pose([x, 0.0, 0.025]),
        size=np.array([0.05, 0.05, 0.05]),
        visible=visible,
        confidence=0.9,
        source=source,
    )


def main(result_path=None):
    metrics = {}
    model = WorldModel()
    model.register(make_object("object_2", x=0.50, visible=False, source="ground_truth"))
    model.register(make_object("object_1", x=0.40, visible=True, source="rgbd_geometry_perception"))

    view = WorldModelViewModel(model)
    assert view.ensure_selection() == "object_1"
    rows = view.rows()
    assert [row.object_id for row in rows] == ["object_1", "object_2"]
    assert rows[0].visible and not rows[1].visible
    assert rows[0].position == (0.4, 0.0, 0.025)
    assert rows[0].size == (0.05, 0.05, 0.05)
    metrics["world_model_rows"] = True

    view.select("object_2")
    assert view.selected_id == "object_2"
    assert view.selected_object().object_id == "object_2"
    metrics["selection"] = True

    model.set_held("object_1")
    held_row = next(row for row in view.rows() if row.object_id == "object_1")
    assert held_row.held
    metrics["held_state"] = True

    before = view.signature()
    model.require("object_1").pose = Pose([0.41, 0.01, 0.025])
    row = next(row for row in view.rows() if row.object_id == "object_1")
    assert row.position == (0.41, 0.01, 0.025)
    assert view.signature() != before
    metrics["live_world_state"] = True

    print("=== WORLD MODEL VIEW UNIT ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("=============================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    args = parser.parse_args()
    try:
        main(args.result)
    except Exception as exc:
        fail_result(args.result, exc)
        sys.exit(1)

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.manipulation.grasping import (
    GraspFailureReason as Reason, GraspSafetyConfig, ParallelJawGripperSpec, SimplePrimitiveGraspPlanner,
)
from robot_skill_stack.manipulation.grasping.primitive import _top_q
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry as Geometry, PrimitiveShape as Shape
from robot_skill_stack.world.model.world_model import WorldModel


def solid(kind="cylinder", *, length=.10, radius=.02, yaw=0., horizontal=False, size=(.05, .05, .05)):
    if kind == "cube":
        g = Geometry(Shape.CUBE, 1, box_size=size, yaw=yaw)
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        extent = np.array([c*size[0]+s*size[1], s*size[0]+c*size[1], size[2]])
    elif kind == "sphere":
        g, extent = Geometry(Shape.SPHERE, 1, radius=radius), np.full(3, 2*radius)
    else:
        axis = np.array([np.cos(yaw), np.sin(yaw), 0]) if horizontal else np.array([0., 0., 1.])
        g = Geometry(Shape.CYLINDER, 1, radius=radius, length=length, axis=axis)
        extent = np.abs(axis)*length+2*radius*np.sqrt(1-axis*axis)
    return WorldObject("target", kind, Pose([.45, 0, extent[2]/2]), extent, geometry=g, visible=True)


class GraspClearanceTests(unittest.TestCase):
    def planner(self, **kwargs):
        return SimplePrimitiveGraspPlanner(support_plane_z=0., **kwargs)

    def good(self, obj, **kwargs):
        planner = self.planner(**kwargs)
        result = planner.plan(obj)
        self.assertTrue(result.ok, (result.failure_reason, result.message, result.details))
        d = result.details
        self.assertGreaterEqual(d["palm_clearance_m"]+1e-9, planner.safety.clearance)
        self.assertGreaterEqual(d["support_clearance_m"]+1e-9, planner.safety.clearance)
        self.assertGreaterEqual(result.plan.pre_grasp.position[2]-planner.gripper.fingertip_offset,
                                d["object_top_z_m"]+planner.safety.approach_clearance-1e-9)
        return result

    def test_tall_upright_cylinder_grasps_upper_sides_not_center(self):
        result = self.good(solid())
        self.assertAlmostEqual(result.plan.grasp.position[2], .075)
        self.assertAlmostEqual(result.details["grasp_height_shift_m"], .025)
        self.assertGreater(result.details["contact_band_z_m"][0], .05)

    def test_normal_cube_preserves_working_center_height(self):
        result = self.good(solid("cube", yaw=.52))
        self.assertAlmostEqual(result.plan.grasp.position[2], .025)
        closing = quat_matrix(result.plan.grasp.orientation)[:, 1]
        self.assertAlmostEqual(abs(closing @ [-np.sin(.52), np.cos(.52), 0]), 1)

    def test_tall_box_uses_upper_side_contact(self):
        result = self.good(solid("cube", size=(.04, .04, .12)))
        self.assertAlmostEqual(result.plan.grasp.position[2], .095)

    def test_sphere_keeps_equator_inside_contact_band(self):
        obj = solid("sphere", radius=.025)
        result = self.good(obj)
        lo, hi = result.details["contact_band_z_m"]
        self.assertLess(lo, obj.pose.position[2])
        self.assertGreater(hi, obj.pose.position[2])
        self.assertIsNone(obj.geometry.yaw)

    def test_lying_cylinder_closes_perpendicular_to_axis(self):
        obj = solid(horizontal=True, yaw=.63)
        result = self.good(obj)
        closing = quat_matrix(result.plan.grasp.orientation)[:, 1]
        self.assertAlmostEqual(closing @ obj.geometry.axis, 0)
        self.assertAlmostEqual(result.details["required_width_m"], .04)
        self.assertAlmostEqual(result.plan.grasp.position[2], .02)

    def test_long_horizontal_cylinder_is_not_rejected_for_length(self):
        self.good(solid(horizontal=True, length=.30))

    def test_height_limit_is_vertical_not_longest_dimension(self):
        for obj in (solid(length=.21), solid("cube", size=(.04, .04, .21))):
            with self.subTest(shape=obj.class_name):
                result = self.planner().plan(obj)
                self.assertEqual(result.failure_reason, Reason.OBJECT_TOO_TALL)
                self.assertIsNone(result.plan)

    def test_palm_contact_depth_rejects_sphere_even_when_width_fits(self):
        obj = solid("sphere", radius=.034)
        result = self.planner().plan(obj)
        self.assertLess(2*obj.geometry.radius, self.planner().gripper.max_width)
        self.assertEqual(result.failure_reason, Reason.INSUFFICIENT_CLEARANCE)

    def test_thin_object_cannot_force_fingertips_through_table(self):
        result = self.planner().plan(solid("cube", size=(.04, .04, .01)))
        self.assertEqual(result.failure_reason, Reason.INSUFFICIENT_CLEARANCE)

    def test_wide_primitives_rejected(self):
        objects = (solid("sphere", radius=.05), solid(radius=.05),
                   solid(horizontal=True, radius=.05), solid("cube", size=(.09, .09, .05)))
        for obj in objects:
            with self.subTest(shape=obj.geometry):
                self.assertEqual(self.planner().plan(obj).failure_reason, Reason.OBJECT_TOO_LARGE)

    def test_open_jaw_side_margin_is_checked_separately(self):
        g = ParallelJawGripperSpec(.075, open_width=.075)
        result = self.planner(gripper=g).plan(solid(radius=.034))
        self.assertEqual(result.failure_reason, Reason.INSUFFICIENT_CLEARANCE)
        self.assertIn("side clearance", result.message)

    def test_thin_and_short_contact_surfaces_rejected(self):
        result = self.planner().plan(solid(radius=.001))
        self.assertEqual(result.failure_reason, Reason.OBJECT_TOO_SMALL)
        result = self.planner().plan(solid(horizontal=True, length=.004))
        self.assertEqual(result.failure_reason, Reason.OBJECT_TOO_SMALL)

    def test_rectangular_box_selects_feasible_face_pair(self):
        obj = solid("cube", size=(.12, .04, .05), yaw=.41)
        result = self.good(obj)
        self.assertAlmostEqual(result.details["required_width_m"], .04)
        self.assertAlmostEqual(quat_matrix(result.plan.grasp.orientation)[:, 1] @ [np.cos(.41), np.sin(.41), 0], 0)

    def test_nearest_equivalent_orientation_is_preferred(self):
        for obj in (solid("cube"), solid("sphere"), solid()):
            with self.subTest(shape=obj.class_name):
                ee = Pose([.4, 0, .3], _top_q(np.pi/2))
                result = self.good(obj, current_pose=lambda: ee)
                self.assertAlmostEqual(abs(np.dot(result.plan.grasp.orientation, ee.orientation)), 1)

    def test_alternate_orientation_can_avoid_known_obstacle(self):
        obj = solid("sphere", radius=.025)
        wm = WorldModel(); wm.register(obj)
        wm.register(WorldObject("neighbor", pose=Pose([.45, .052, .015]), size=[.016, .016, .03], visible=True))
        result = self.good(obj, world_model=wm)
        self.assertTrue(result.details["rejected_candidates"])
        self.assertIn("neighbor", result.details["checked_object_ids"])
        self.assertGreater(result.details["orientation_change_rad"], .1)

    def test_overhead_obstacle_blocks_every_candidate(self):
        obj = solid(); wm = WorldModel(); wm.register(obj)
        wm.register(WorldObject("shelf", pose=Pose([.45, 0, .18]), size=[.20, .20, .01], visible=True))
        result = self.planner(world_model=wm).plan(obj)
        self.assertEqual(result.failure_reason, Reason.APPROACH_BLOCKED)
        self.assertEqual(result.details["blocking_object_id"], "shelf")

    def test_object_lift_sweep_is_checked_not_only_hand(self):
        obj = solid(horizontal=True, length=.20); wm = WorldModel(); wm.register(obj)
        wm.register(WorldObject("overhang", pose=Pose([.53, 0, .09]), size=[.01, .01, .01], visible=True))
        result = self.planner(world_model=wm).plan(obj)
        self.assertEqual(result.failure_reason, Reason.APPROACH_BLOCKED)
        self.assertEqual(result.details["blocked_part"], "held_object_lift")

    def test_cached_obstacle_is_conservative_until_forgotten(self):
        obj = solid(); wm = WorldModel(); wm.register(obj)
        wm.register(WorldObject("cached", pose=Pose([.45, 0, .18]), size=[.2, .2, .01], visible=False))
        self.assertEqual(self.planner(world_model=wm).plan(obj).failure_reason, Reason.APPROACH_BLOCKED)
        wm.remove("cached")
        self.good(obj, world_model=wm)

    def test_visible_unknown_geometry_does_not_mean_free_space(self):
        obj = solid(); wm = WorldModel(); wm.register(obj)
        wm.register(WorldObject("unknown", pose=Pose([.4, 0, .1]), visible=True))
        result = self.planner(world_model=wm).plan(obj)
        self.assertEqual(result.failure_reason, Reason.APPROACH_BLOCKED)
        self.assertEqual(result.details["blocked_part"], "unknown_obstacle_geometry")

    def test_reachability_filters_candidates_and_preserves_alternatives(self):
        desired = _top_q(np.pi/2)
        reachable = lambda p: abs(np.dot(desired, p.orientation)) > .999
        self.good(solid("cube"), pose_reachable=reachable)
        result = self.planner(pose_reachable=lambda p: False).plan(solid())
        self.assertEqual(result.failure_reason, Reason.NO_FEASIBLE_GRASP)

    def test_unsupported_tilts_are_not_silently_upright(self):
        obj = solid(); obj.geometry.axis = np.array([1., 0, 1.])/np.sqrt(2)
        self.assertEqual(self.planner().plan(obj).failure_reason, Reason.UNSUPPORTED_ORIENTATION)
        obj = solid("cube"); obj.pose.orientation = np.array([np.cos(.3), np.sin(.3), 0, 0])
        self.assertEqual(self.planner().plan(obj).failure_reason, Reason.UNSUPPORTED_ORIENTATION)

    def test_malformed_geometry_is_structured_failure(self):
        for field, value in (("radius", None), ("radius", -1), ("radius", np.nan),
                             ("length", np.inf), ("length", 0), ("axis", [0, 0, 0]), ("axis", [np.nan, 0, 1])):
            with self.subTest(field=field, value=value):
                obj = solid(); setattr(obj.geometry, field, value)
                result = self.planner().plan(obj)
                self.assertEqual(result.failure_reason, Reason.INVALID_GEOMETRY)
        for dims in ([1, 2], [0, .04, .05], [.05, np.nan, .05]):
            obj = solid("cube"); obj.geometry.box_size = np.array(dims)
            self.assertEqual(self.planner().plan(obj).failure_reason, Reason.INVALID_GEOMETRY)
        obj = solid("cube"); obj.geometry.yaw = None
        self.assertEqual(self.planner().plan(obj).failure_reason, Reason.INVALID_GEOMETRY)

    def test_custom_settings_are_validated(self):
        for args in ({"clearance": -1}, {"max_object_height": np.inf}, {"min_contact_overlap": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                GraspSafetyConfig(**args)
        for args in ({"max_width": np.nan}, {"open_width": .01}, {"tcp_to_palm": 0}, {"pad_below_tcp": .1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ParallelJawGripperSpec(**{"max_width": .075, **args})
        with self.assertRaises(ValueError):
            self.planner(grasp_orientation=[1, 0, 0, 0])

    def test_config_defaults_and_overrides_do_not_require_scene_replacement(self):
        original = '[robot]\nid="franka"\ntype="franka"\nprim_path="/Franka"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scene.toml"
            path.write_text(original)
            cfg = load_scene_config(path)
            self.assertEqual(cfg.grasping.gripper.open_width, .08)
            path.write_text(original+'\n[grasping.safety]\nmax_object_height = 0.15\n'
                            '[grasping.gripper]\ntcp_to_palm = 0.025\n')
            changed = load_scene_config(path)
            self.assertEqual(changed.grasping.safety.max_object_height, .15)
            self.assertEqual(changed.grasping.gripper.tcp_to_palm, .025)
            self.assertEqual(changed.perception.resolution, cfg.perception.resolution)
            np.testing.assert_equal(changed.perception.discovery.workspace_min, cfg.perception.discovery.workspace_min)

    def test_many_accepted_plans_satisfy_height_and_opening_constraints(self):
        count = 0
        for radius in (.012, .02, .025, .03):
            for length in (.025, .05, .10, .18):
                for horizontal in (False, True):
                    obj = solid(radius=radius, length=length, horizontal=horizontal, yaw=.52)
                    result = self.planner().plan(obj)
                    if result.ok:
                        self.good(obj)
                        count += 1
                    else:
                        self.assertIsNotNone(result.failure_reason)
        self.assertGreater(count, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)

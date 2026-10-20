"""Physical scene/layout and visibility-scoring checks; no holdout generation."""
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from tests.support.multiview_final_research import (
    CameraLayout, FinalScene, REFERENCE_WORLD_FROM_CAMERAS, SEEDS, VISIBILITY,
    WORKSPACE_MAX, WORKSPACE_MIN, audit_final_scene, balanced_layout_order,
    camera_layouts, final_scenes, fixture_geometry_errors, layout_digest,
    scene_frame_content, scene_frame_count, workspace_axes,
)
from tests.support.multiview_iteration2 import box, footprint_half_extent, sphere
from tests.support.primitive_scene import Solid, intersect


def ideal_final_frames(scene, layout, index=3, resolution=(320, 240)):
    """Analytic native-lens rays; inherited robot occlusion needs native audit."""
    objects, _ = scene_frame_content(scene, index)
    solids = [Solid(obj.shape, tuple(obj.center), tuple(obj.size), obj.yaw,
                    obj.axis, obj.radius, obj.length) for obj in objects]
    width, height = resolution
    y, x = np.indices((height, width))
    frames = []
    for ci, (eye, target) in enumerate(zip(layout.eyes, layout.targets)):
        # Native capture confirms the two inherited USD camera lenses differ.
        fx, fy = ((554.2562255859375, 554.2562866210938),
                  (732.999267578125, 732.999267578125))[ci]
        k = np.array([[fx*width/640, 0., width/2],
                      [0., fy*height/480, height/2], [0., 0., 1.]])
        rays = np.c_[(x.ravel()+.5-k[0, 2])/k[0, 0],
                     (y.ravel()+.5-k[1, 2])/k[1, 1], np.ones(height*width)]
        fixed = None if layout.world_from_cameras is None else layout.world_from_cameras[ci]
        if fixed is not None:
            transform = np.asarray(fixed, float).copy()
        else:
            forward = np.asarray(target)-eye; forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
            transform[:3, 3] = eye
        world_rays, origin = rays @ transform[:3, :3].T, transform[:3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = -origin[2]/world_rays[:, 2]
        depth = np.where(depth > 0, depth, np.inf)
        for solid in solids:
            depth = np.minimum(depth, intersect(solid, origin, world_rays))
        frames.append(DepthFrame(str(ci), (index+1, (index+1)*.1), depth.reshape(height, width).astype(np.float32),
                                 k.copy(), transform, .1*(index+1)))
    return tuple(frames)


class FinalResearchTests(unittest.TestCase):
    def test_inspected_reference_and_front_workspace_axes(self):
        origin, forward, left, up = workspace_axes()
        np.testing.assert_array_equal(origin, [0, 0, 0])
        np.testing.assert_array_equal(forward, [1, 0, 0])
        np.testing.assert_array_equal(left, [0, 1, 0])
        np.testing.assert_array_equal(up, [0, 0, 1])
        layouts = camera_layouts()
        self.assertEqual(len(layouts), 10)
        self.assertEqual(len({layout_digest(layout) for layout in layouts}), 10)
        self.assertEqual(layouts[0].name, 'original_reference')
        self.assertEqual(layouts[0].world_from_cameras, REFERENCE_WORLD_FROM_CAMERAS)
        np.testing.assert_allclose(layouts[0].eyes[0], [1.65896, -.05281, 1.10358], atol=4e-6)
        np.testing.assert_array_equal(layouts[0].eyes[1], [.25, -.85, .65])
        for layout in layouts:
            for eye in layout.eyes:
                self.assertFalse(np.all(np.asarray(eye)[:2] >= WORKSPACE_MIN[:2]) and
                                 np.all(np.asarray(eye)[:2] <= WORKSPACE_MAX[:2]))
                self.assertGreater(eye[2], WORKSPACE_MAX[2])

    def test_balanced_order_preserves_each_fixed_layout(self):
        layouts = camera_layouts()
        first = []
        for scene_index in range(10):
            ordered = balanced_layout_order(layouts, scene_index)
            self.assertEqual({item.name for item in ordered}, {item.name for item in layouts})
            first.append(ordered[0].name)
        self.assertEqual(len(set(first)), 10)

    def test_coarse_shortlist_geometry_counts_gaps_and_independence(self):
        # Deliberately do not invoke final_scenes('holdout'), even for a guard
        # or determinism test. Independent data are generated only after freeze.
        self.assertEqual(len(set(SEEDS.values())), 3)
        first, second = final_scenes('coarse'), final_scenes('shortlist')
        self.assertEqual(len(first), 13)
        self.assertEqual(sum(len(scene.objects) for scene in first), 31)
        self.assertEqual(len(second), 15)
        self.assertEqual(sum(len(scene.objects) for scene in second), 37)
        self.assertTrue({scene.seed for scene in first}.isdisjoint(scene.seed for scene in second))
        for scenes in (first, second):
            for scene in scenes:
                for index in range(scene_frame_count(scene)):
                    objects, _ = scene_frame_content(scene, index)
                    self.assertEqual(fixture_geometry_errors(objects), [], (scene.key, index))
                pairs = list(zip(scene.objects[::2], scene.objects[1::2])) if scene.name == 'three_pairs_gap8' else (
                    [(scene.objects[0], scene.objects[1])] if scene.requirements and 'gap_mm' in scene.requirements else [])
                for a, b in pairs:
                    if a.shape == b.shape == 'sphere':
                        gap = np.linalg.norm(a.center-b.center)-a.radius-b.radius
                    elif b.shape == 'sphere':
                        yaw = a.yaw; rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                        gap = np.linalg.norm(np.maximum(abs((b.center-a.center) @ rotation)-a.size/2, 0))-b.radius
                    else:
                        direction = b.center[:2]-a.center[:2]; direction /= np.linalg.norm(direction)
                        gap = np.linalg.norm(b.center[:2]-a.center[:2])-sum(footprint_half_extent(obj, direction) for obj in (a, b))
                    self.assertAlmostEqual(gap*1000, scene.requirements['gap_mm'], places=7)

    def test_temporal_motion_preserves_all_physical_objects_with_bounded_steps(self):
        for split, count in (('coarse', 3), ('shortlist', 4)):
            scene = next(scene for scene in final_scenes(split) if scene.temporal)
            previous, names = None, [obj.name for obj in scene.objects]
            self.assertEqual(len(names), count)
            for index in range(12):
                objects, phase = scene_frame_content(scene, index)
                self.assertEqual([obj.name for obj in objects], names)
                self.assertEqual(phase, ('clear', 'moving', 'covered', 'revealed')[index//3])
                if previous is not None:
                    self.assertLess(max(np.linalg.norm(a.center-b.center) for a, b in zip(objects, previous)), .12)
                previous = objects
            if split == 'coarse':
                self.assertNotIn('fixture_version', scene.requirements)
                covered, _ = scene_frame_content(scene, 6)
                expected = scene.objects[0].center[:2]+np.asarray(scene.requirements['movement_delta_m'])[:2]+[.105, 0.]
                np.testing.assert_array_equal(covered[2].center[:2], expected)
            else:
                self.assertEqual(scene.requirements['fixture_version'], 2)
                self.assertEqual([item['object_index'] for item in scene.requirements['cover_destinations_m']], [2, 3])

    def test_added_supported_occlusion_has_one_view_and_full_hiding_in_analytic_depth(self):
        scenes = {scene.name: scene for scene in final_scenes('shortlist')}
        for name in ('asymmetric_high_left', 'front_symmetric', 'front_wider'):
            layout = next(layout for layout in camera_layouts() if layout.name == name)
            for scene_name, both in (('supported_one_view', False), ('supported_full_hide', True)):
                scene = scenes[scene_name]
                self.assertEqual(len(scene.objects), 3 if both else 2)
                self.assertEqual(fixture_geometry_errors(scene.objects), [])
                audit = audit_final_scene(scene, ideal_final_frames(scene, layout, resolution=(640, 480)))
                self.assertTrue(audit['valid'], audit['errors'])
                raw = [camera['raw_points'] for camera in audit['objects'][0]['cameras']]
                self.assertEqual(raw[0], 0, (name, scene_name, raw))
                if both:
                    self.assertEqual(raw[1], 0, (name, raw))
                else:
                    self.assertGreaterEqual(raw[1], VISIBILITY['minimum_camera_surface_points'])
                for cover in audit['objects'][1:]:
                    self.assertGreaterEqual(sum(camera['raw_points'] for camera in cover['cameras']),
                                            VISIBILITY['minimum_camera_surface_points'])

    def test_two_cover_temporal_depth_restores_same_physical_target(self):
        scene = next(scene for scene in final_scenes('shortlist') if scene.temporal)
        layout = next(layout for layout in camera_layouts() if layout.name == 'asymmetric_high_left')
        names = [obj.name for obj in scene.objects]
        for index, hidden in ((2, False), (6, True), (11, False)):
            objects, _ = scene_frame_content(scene, index)
            self.assertEqual([obj.name for obj in objects], names)
            self.assertEqual(fixture_geometry_errors(objects), [])
            audit = audit_final_scene(scene, ideal_final_frames(scene, layout, index=index,
                resolution=(640, 480)), index=index)
            self.assertTrue(audit['valid'], audit['errors'])
            target = audit['objects'][0]
            self.assertEqual(target['fully_hidden'], hidden)
            raw = [camera['raw_points'] for camera in target['cameras']]
            if hidden:
                self.assertEqual(raw, [0, 0])
            else:
                self.assertTrue(all(value >= VISIBILITY['minimum_camera_surface_points'] for value in raw))

    def test_visibility_is_physical_surface_fraction_and_does_not_invalidate_occlusion(self):
        layout = next(layout for layout in camera_layouts() if layout.name == 'front_narrower')
        clear = FinalScene('clear', 'coarse', 1, (sphere('target', (.40, 0.), .018),))
        hidden = FinalScene('hidden', 'coarse', 1, clear.objects+(box('supported_cover', (.50, 0.), (.10, .15, .15)),))
        clear_audit = audit_final_scene(clear, ideal_final_frames(clear, layout, resolution=(640, 480)))
        hidden_audit = audit_final_scene(hidden, ideal_final_frames(hidden, layout, resolution=(640, 480)))
        self.assertTrue(clear_audit['valid'])
        self.assertTrue(clear_audit['objects'][0]['sufficiently_visible'])
        self.assertGreater(clear_audit['objects'][0]['union_visible_surface_fraction'], VISIBILITY['minimum_surface_fraction'])
        self.assertTrue(hidden_audit['valid'], hidden_audit['errors'])
        self.assertTrue(hidden_audit['objects'][0]['fully_hidden'])
        self.assertFalse(hidden_audit['objects'][0]['sufficiently_visible'])
        cube = FinalScene('cube', 'coarse', 2, (box('target', (.45, 0.), (.05, .04, .045), .25),))
        audit = audit_final_scene(cube, ideal_final_frames(cube, layout, resolution=(640, 480)))
        self.assertEqual(set(audit['objects'][0]['faces']), {'-x', '+x', '-y', '+y', '+z'})
        self.assertTrue(all(0 <= face['union_visible_fraction'] <= 1 for face in audit['objects'][0]['faces'].values()))


if __name__ == '__main__':
    unittest.main()

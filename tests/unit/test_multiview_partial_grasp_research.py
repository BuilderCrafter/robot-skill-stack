"""Independent ideal-depth checks for the separate physical-occlusion ablation."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth
from scripts.benchmark_multiview_partial_grasp import ROOT, audit_capture, local_extent, partial_scenes
from scripts.benchmark_multiview_robustness import capture_workspace, save_frames, write_json
from tests.support.multiview_iteration2 import surface_memberships
from tests.support.multiview_readiness_research import EYES, make_planner, truth_world_object
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


class PartialGraspResearchTests(unittest.TestCase):
    def test_static_fixture_rays_clip_extents_and_keep_independent_controls(self):
        cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
        measured = {}
        scenes = partial_scenes()
        planner = make_planner(WorldModel())
        for scene in scenes:
            target = scene.objects[0]
            self.assertFalse(planner.plan(truth_world_object(target)).ok, scene.name)
            solids = scene.objects+scene.clutter
            rendered = [Solid(obj.shape, tuple(obj.center), tuple(obj.size), obj.yaw) for obj in solids]
            views = []
            for frame in analytic_frames(rendered, eyes=EYES['selection']):
                cloud = crop_workspace(project_depth(frame, stride=1)[0], cfg.scene)
                selected = cloud[surface_memberships(cloud, solids) == 0]
                self.assertGreaterEqual(len(selected), 30, scene.name)
                views.append(local_extent(selected, target)['span_mm'])
            measured[scene.name] = np.asarray(views)
        full = measured['partial_box_reference_90']
        self.assertTrue((full[:, :2] > 85).all())
        self.assertGreater(full[0, 1]-measured['partial_box_A'][0, 1], 10)
        self.assertLess(np.max(abs(full[1, :2]-measured['partial_box_A'][1, :2])), 5)
        self.assertGreater(full[1, 0]-measured['partial_box_B'][1, 0], 10)
        self.assertLess(np.max(abs(full[0, :2]-measured['partial_box_B'][0, :2])), 5)
        self.assertTrue((full[:, 1]-measured['partial_box_both'][:, 1] > 10).all())

    def test_native_file_audit_preserves_unobserved_reference_as_invalid(self):
        cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
        scenes = partial_scenes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = []
            for scene in scenes:
                solids = scene.objects+scene.clutter
                rendered = [Solid(obj.shape, tuple(obj.center), tuple(obj.size), obj.yaw) for obj in solids]
                frames = analytic_frames(rendered, eyes=EYES['selection'])
                snapshots = []
                for index in (2, 3):
                    files = save_frames(output, scene.key, index, frames)
                    capture_workspace(output, scene.key, index, frames, scene.objects, cfg.scene)
                    snapshots.append(dict(index=index, files=files))
                records.append(dict(key=scene.key, frames=snapshots))
            write_json(output/'capture_manifest.json', dict(scenes=records))
            write_json(output/'rows.json', [dict(truth_feasible=False)])
            self.assertTrue(audit_capture(output, scenes)['valid'])
            reference = next(row for row in records if row['key'].endswith('reference_90'))
            for snapshot in reference['frames']:
                for filename in snapshot['files']:
                    path = output/filename
                    with np.load(path, allow_pickle=False) as saved:
                        values = {name: saved[name].copy() for name in saved.files}
                    values['depth'][:] = np.nan
                    np.savez_compressed(path, **values)
            audit = audit_capture(output, scenes)
            self.assertFalse(audit['valid'])
            self.assertTrue(any(row['reason'] == 'missing_or_unobserved_90mm_reference'
                                for row in audit['failures']))


if __name__ == '__main__':
    unittest.main()

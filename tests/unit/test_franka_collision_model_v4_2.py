"""Solid-cell containment, frame alignment and frozen artifact contracts."""
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.generate_franka_collision_model_v4_2 import (
    CookedPart, FINGER_RANGE, certify_cell, clip_convex_vertices, convex_cells,
    cooked_body_vertices, enclosing_sphere, finger_transforms, generate,
)


ROOT = Path(__file__).resolve().parents[2]


def box(low, high):
    return np.array(list(itertools.product(*zip(low, high))), dtype=float)


def finger_urdf(path):
    path.write_text('<robot>'+''.join(
        f'<joint name="panda_finger_joint{index}" type="prismatic">'
        '<parent link="panda_hand"/>'
        f'<child link="panda_{side}finger"/><origin xyz="0 0 .0584" rpy="0 0 0"/>'
        f'<axis xyz="0 {direction} 0"/></joint>'
        for index, side, direction in ((1, 'left', 1), (2, 'right', -1)))+'</robot>')
    return path


class FrankaCollisionModelTests(unittest.TestCase):
    def test_clipped_cell_retains_all_required_plane_edge_intersections(self):
        points = box([-1, -1, -1], [1, 1, 1])
        clipped = clip_convex_vertices(points, [1., 0, 0], .25, upper=True)
        self.assertTrue(np.all(clipped[:, 0] <= .25+1e-12))
        self.assertEqual(len(clipped[clipped[:, 0] == .25]), 8)
        # Face triangulation introduces harmless diagonal intersections. The
        # original face corners, needed for the full cross section, remain.
        for y, z in itertools.product((-1., 1.), repeat=2):
            self.assertTrue(np.any(np.all(clipped == [.25, y, z], axis=1)))
        self.assertIsNone(clip_convex_vertices(points, [1., 0, 0], 2., upper=False))

    def test_rotated_partition_covers_original_solid_volume(self):
        from scipy.spatial import ConvexHull
        points = box([-.07, -.04, -.03], [.07, .04, .03])
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        points = points@rotation.T+[.1, -.2, .3]
        cells, axes = convex_cells(points, (.060, .050))
        self.assertEqual(len(cells), 6)
        original = ConvexHull(points)
        self.assertAlmostEqual(sum(ConvexHull(cell).volume for cell in cells), original.volume, places=12)
        for cell in cells:
            self.assertLessEqual(np.max(cell@original.equations[:, :3].T+original.equations[:, 3]), 1e-11)
            self.assertTrue(np.all(np.ptp(cell@axes, axis=0)[:2] <= [.060+1e-11, .050+1e-11]))

    def test_coverage_requires_one_sphere_for_a_whole_cell(self):
        tetrahedron = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0], [0., 0., 1.]])
        # Four tiny vertex-centered spheres cover every vertex, yet leave the
        # tetrahedron interior uncovered. No individual sphere certifies it.
        for center in tetrahedron:
            with self.assertRaisesRegex(ValueError, 'one sphere'):
                certify_cell(tetrahedron, center, .01)
        center, radius = enclosing_sphere(tetrahedron)
        proof = certify_cell(tetrahedron, center, radius)
        self.assertGreater(proof['minimum_containment_margin_m'], 0)
        with self.assertRaises(ValueError):
            certify_cell(tetrahedron, center, radius-.1)

    def test_mesh_scale_and_translation_are_applied_in_body_frame(self):
        points = box([0., 0, 0], [2., 4., 6.])
        transform = np.eye(4)
        transform[:3, :3] *= .01
        transform[:3, 3] = [.1, .2, .3]
        converted = cooked_body_vertices(points, transform)
        np.testing.assert_allclose(converted.min(0), [.1, .2, .3])
        np.testing.assert_allclose(converted.max(0), [.12, .24, .36])
        transform[0, 0] = 0
        with self.assertRaises(ValueError):
            cooked_body_vertices(points, transform)

    def test_finger_sweep_uses_actual_urdf_directions_and_positive_separation(self):
        with TemporaryDirectory() as folder:
            urdf = finger_urdf(Path(folder)/'robot.urdf')
            transforms = finger_transforms(urdf, .003)
            np.testing.assert_allclose(transforms['panda_leftfinger'][0][:3, 3], [0, .003, .0584])
            np.testing.assert_allclose(transforms['panda_rightfinger'][0][:3, 3], [0, -.003, .0584])
            left = box([-.01, 0, 0], [.01, .02, .05])
            right = box([-.01, -.02, 0], [.01, 0, .05])
            parts = [CookedPart('panda_link0', 0, box([-.1, -.1, 0], [.1, .1, .1])),
                     CookedPart('panda_leftfinger', 1, left), CookedPart('panda_rightfinger', 2, right)]
            stock = dict(cspace_to_urdf_rules=[dict(name=f'panda_finger_joint{index}', value=.025)
                                              for index in (1, 2)], collision_spheres=[{'panda_hand': []}])
            descriptor, metadata = generate(stock, parts, urdf)
            self.assertAlmostEqual(metadata['minimum_physical_finger_separating_projection_m'], .002)
            self.assertTrue(all(rule['value'] == .003 for rule in descriptor['cspace_to_urdf_rules']))
            groups = {name: rows for entry in descriptor['collision_spheres'] for name, rows in entry.items()}
            for certificate, vertices in zip(metadata['solid_coverage']['certificates'], (left, right)):
                transform, axis = transforms[certificate['body']]
                spheres = groups[certificate['body']]
                for opening in (FINGER_RANGE[0], .0021, .003, .0047, FINGER_RANGE[1]):
                    # The planner frame is fixed at the midpoint. Actual
                    # physical translation relative to that frame is bounded.
                    actual = vertices+(opening-.003)*axis
                    distances = np.array([np.linalg.norm(actual-row['center'], axis=1)-row['radius']
                                          for row in spheres])
                    self.assertLess(np.max(np.min(distances, axis=0)), 0)

    def test_frozen_descriptor_and_certificate_remain_consistent(self):
        import yaml
        descriptor_path = ROOT/'config/motion/franka_enhanced_v4_2.yaml'
        metadata = json.loads(descriptor_path.with_suffix('.json').read_text())
        document = yaml.safe_load(descriptor_path.read_text())
        self.assertEqual(hashlib.sha256(descriptor_path.read_bytes()).hexdigest(), metadata['descriptor_sha256'])
        groups = {name: rows for entry in document['collision_spheres'] for name, rows in entry.items()}
        self.assertEqual(sum(map(len, groups.values())), 102)
        self.assertEqual(metadata['added_sphere_count'], 47)
        self.assertEqual(metadata['solid_coverage']['moving_body_count'], 10)
        self.assertEqual(metadata['solid_coverage']['convex_cell_count'], 47)
        self.assertEqual(metadata['measured_frame_and_numerical_allowance_m'], 1e-5)
        self.assertLess(metadata['solid_coverage']['maximum_partition_relative_volume_error'], 1e-8)
        self.assertGreater(metadata['minimum_physical_finger_separating_projection_m'], .0019)
        self.assertEqual(len(groups['panda_leftfinger']), 3)
        self.assertEqual(len(groups['panda_rightfinger']), 3)
        exclusion, = metadata['certified_self_collision_exclusions']
        self.assertEqual(set(exclusion['frames']), {'panda_leftfinger', 'panda_rightfinger'})
        self.assertGreater(exclusion['minimum_separating_projection_m'], exclusion['numerical_tolerance_m'])
        for body in metadata['solid_coverage']['certificates']:
            self.assertGreater(body['original_convex_volume_m3'], 0)
            self.assertAlmostEqual(body['cell_volume_sum_m3']/body['original_convex_volume_m3'], 1., places=10)
            for cell in body['cells']:
                sphere = groups[cell['sphere_frame']][cell['sphere_index_on_frame']]
                self.assertAlmostEqual(sphere['radius']-cell['swept_axis_padding_m'], cell['radius_m'])
                self.assertGreater(cell['minimum_containment_margin_m'], 0)
        base = np.asarray(metadata['base_aabb_corners_m'])
        self.assertEqual(len(np.unique(base, axis=0)), 8)
        np.testing.assert_array_equal(base.min(0), metadata['base_aabb_minimum_m'])
        np.testing.assert_array_equal(base.max(0), metadata['base_aabb_maximum_m'])


if __name__ == '__main__':
    unittest.main()

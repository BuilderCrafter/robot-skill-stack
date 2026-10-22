"""Independent physical reference and denominator tests; no simulator imports."""
import json
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState
from tests.support.spatial_mapping_reliability import (
    ReferenceShape, center_membership, conservative_cell_overlap, conservative_cell_interior, neighborhood_indices,
    score_clear_region, score_surface_events, score_voxel_centers,
)


def transform(center=(0., 0., 0.), yaw=0., scale=(1., 1., 1.)):
    c, s = np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.diag(scale)
    matrix[:3, 3] = center
    return matrix


def box(name='box', center=(0., 0., 0.), size=(.1, .1, .1), yaw=0.):
    return ReferenceShape(name, 'box', transform(center, yaw), dimensions=size)


def wedge():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    triangles = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                          [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    return vertices, triangles


class ReliabilityReferenceTests(unittest.TestCase):
    def test_rotated_box_and_affine_curved_solids_do_not_use_world_aabb_truth(self):
        shape = box(center=(.3, 0., .1), size=(.12, .02, .04), yaw=np.pi/4)
        points = np.array([[.3, 0., .1], [.34, -.04, .1], [.3, 0., .13]])
        low, high = shape.world_bounds()
        self.assertTrue(np.all((points[1] >= low) & (points[1] <= high)))
        np.testing.assert_array_equal(center_membership(points, [shape]), [True, False, False])
        sphere = ReferenceShape('ellipsoid', 'sphere', transform(scale=(2., 1., .5)), radius=1.)
        np.testing.assert_array_equal(center_membership([[1.9, 0., 0.], [0., 1.1, 0.], [0., 0., .5]], [sphere]),
                                      [True, False, True])
        cylinder = ReferenceShape('lying', 'cylinder', transform(scale=(1., .2, .4)),
                                  radius=1., height=2., axis='X')
        np.testing.assert_array_equal(center_membership([[1., 0., 0.], [1.01, 0., 0.], [0., .21, 0.]], [cylinder]),
                                      [True, False, False])

    def test_convex_wedge_membership_and_nonconvex_union_are_actual_solids(self):
        vertices, triangles = wedge()
        shape = ReferenceShape('wedge', 'convex_mesh', np.eye(4), vertices=vertices, triangles=triangles)
        np.testing.assert_array_equal(center_membership([[.2, .2, .5], [.8, .8, .5], [.5, .5, .5]], [shape]),
                                      [True, False, True])
        # The same physical name may consist of several analytic parts.
        parts = [box('L', (.1, .3, .5), (.2, .6, 1.)), box('L', (.4, .1, .5), (.4, .2, 1.))]
        np.testing.assert_array_equal(center_membership([[.1, .5, .5], [.5, .1, .5], [.5, .5, .5]], parts),
                                      [True, True, False])
        with self.assertRaisesRegex(ValueError, 'closed'):
            ReferenceShape('open', 'convex_mesh', np.eye(4), vertices=vertices, triangles=triangles[:-1])
        concave = vertices.copy(); concave[3] = [.4, .4, .4]
        with self.assertRaisesRegex(ValueError, 'convex'):
            ReferenceShape('concave', 'convex_mesh', np.eye(4), vertices=concave, triangles=triangles)

    def test_owned_immutable_payload_roundtrip_and_invalid_reference_rejection(self):
        original = box(yaw=.3)
        clone = ReferenceShape.from_payload(json.loads(json.dumps(original.to_payload())))
        np.testing.assert_array_equal(center_membership([[0., 0., 0.], [.2, .2, .2]], [clone]), [True, False])
        for values in (clone.world_from_local, clone.dimensions, clone._inverse):
            with self.assertRaises(ValueError):
                values.setflags(write=True)
        vertices, triangles = wedge()
        mesh = ReferenceShape('wedge', 'convex_mesh', np.eye(4), vertices=vertices, triangles=triangles)
        for values in (mesh.vertices, mesh.triangles, mesh._normal, mesh._offset):
            with self.assertRaises(ValueError):
                values.setflags(write=True)
        for options in (dict(world_from_local=np.zeros((4, 4))), dict(dimensions=[1., 0., 1.]),
                        dict(reference_role='approximate_aabb')):
            payload = original.to_payload(); payload.update(options)
            with self.assertRaises(ValueError):
                ReferenceShape.from_payload(payload)

    def test_center_boundary_and_conservative_cell_overlap_are_distinct(self):
        spec = GridSpec((0., 0., 0.), (.03, .01, .01), .01)
        thin = box('thin', (.01, .005, .005), (.002, .006, .006))
        indices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(center_membership(spec.indices_to_centers(indices), [thin]), [False]*3)
        np.testing.assert_array_equal(conservative_cell_overlap(spec, indices, [thin]), [True, True, False])
        # The final shortened cell uses its actual midpoint, not +half a nominal voxel.
        partial = GridSpec((0., 0., 0.), (.025, .01, .01), .01)
        last = box('last', (.0225, .005, .005), (.001, .001, .001))
        self.assertTrue(center_membership(partial.indices_to_centers([[2, 0, 0]]), [last])[0])
        neighborhood = neighborhood_indices(spec, [thin], margin=0.)
        np.testing.assert_array_equal(neighborhood, [[0, 0, 0], [1, 0, 0]])

    def test_conservative_overlap_never_discards_sampled_inside_points_for_affine_shapes(self):
        spec = GridSpec((-.2, -.2, -.2), (.2, .2, .2), .02)
        shapes = [box(size=(.12, .025, .04), yaw=.7),
                  ReferenceShape('sphere', 'sphere', transform((.06, .02, .02), scale=(1.4, .5, 1.)), radius=.03),
                  ReferenceShape('cylinder', 'cylinder', transform((-.04, -.03, 0.), .4), radius=.025, height=.08)]
        rng = np.random.default_rng(1241)
        points = rng.uniform(-.15, .15, (1000, 3))
        inside = center_membership(points, shapes)
        self.assertGreater(inside.sum(), 0)
        self.assertTrue(conservative_cell_overlap(spec, spec.world_to_indices(points[inside]), shapes).all())

    def test_whole_cell_interior_requires_all_corners_inside_the_same_convex_part(self):
        spec = GridSpec((0., 0., 0.), (.1, .1, .1), .1)
        whole = box(center=(.05, .05, .05), size=(.1, .1, .1))
        self.assertTrue(conservative_cell_interior(spec, [[0, 0, 0]], [whole])[0])
        thin = box(center=(.05, .05, .05), size=(.006, .1, .1))
        self.assertTrue(center_membership([[.05]*3], [thin])[0])
        self.assertFalse(conservative_cell_interior(spec, [[0, 0, 0]], [thin])[0])
        # Every corner belongs to the union, but the central slab is empty.
        separate = [box('part', (0., .05, .05), (.04, .1, .1)),
                    box('part', (.1, .05, .05), (.04, .1, .1))]
        self.assertFalse(conservative_cell_interior(spec, [[0, 0, 0]], separate)[0])
        partial = GridSpec((0., 0., 0.), (.025, .01, .01), .01)
        exact_last = box(center=(.0225, .005, .005), size=(.005, .01, .01))
        self.assertTrue(conservative_cell_interior(partial, [[2, 0, 0]], [exact_last])[0])

    def test_voxel_metrics_keep_false_free_false_occupied_unknown_and_excess_denominators(self):
        spec = GridSpec((0., 0., 0.), (.5, .1, .1), .1)
        occupancy = OccupancyMap(spec, clock=lambda: 100.)
        for stamp in (98., 99.):
            occupancy.integrate([[1, 0, 0]], [[2, 0, 0], [3, 0, 0]], timestamp=stamp, now=100.)
        occupancy.integrate([[0, 0, 0]], [], timestamp=99.5, now=100.)
        shape = box(center=(.15, .05, .05), size=(.1, .1, .1))
        indices = np.array([[i, 0, 0] for i in range(5)])
        result = score_voxel_centers(occupancy.snapshot(), [shape], indices)
        self.assertEqual(result['reference_occupied'], dict(voxels=1, unknown=0, free=1, occupied=0))
        self.assertEqual(result['false_free'], dict(count=1, denominator=1, rate=1.))
        self.assertEqual(result['false_occupied'], dict(count=2, denominator=4, rate=.5))
        self.assertEqual(result['unknown'], dict(count=2, denominator=5, rate=.4))
        self.assertEqual(result['definite_occupied_excess'], dict(count=1, denominator=2, rate=.5))
        self.assertEqual(result['center_free_possible_cell_overlap']['occupied'], 1)
        self.assertEqual(result['observed_region']['voxels'], 4)
        self.assertEqual(result['observed_region']['unknown'], 1)
        self.assertEqual(result['observed_region']['agreement_classified']['denominator'], 3)

    def test_motion_truth_disagreement_is_not_counted_as_false_free(self):
        spec = GridSpec((0., 0., 0.), (.5, .1, .1), .1)
        occupancy = OccupancyMap(spec, clock=lambda: 100.)
        for stamp in (98., 99.):
            occupancy.integrate([[1, 0, 0], [3, 0, 0]], [], timestamp=stamp, now=100.)
        first = [box('moving', (.15, .05, .05), (.1, .1, .1))]
        second = [box('moving', (.35, .05, .05), (.1, .1, .1))]
        result = score_voxel_centers(occupancy.snapshot(), first,
            [[i, 0, 0] for i in range(5)], other_exposures=(second,))
        self.assertEqual(result['motion_disagreement']['voxels'], 2)
        self.assertEqual(result['motion_disagreement']['free'], 2)
        self.assertEqual(result['false_free'], dict(count=0, denominator=0, rate=None))
        self.assertEqual(result['reference_free']['voxels'], 3)
        self.assertEqual(result['observed_region']['voxels'], 0)

    def test_surface_event_and_unique_voxel_denominators_remain_separate(self):
        spec = GridSpec((0., 0., 0.), (.3, .1, .1), .1)
        occupancy = OccupancyMap(spec, clock=lambda: 100.)
        for stamp in (98., 99.):
            occupancy.integrate([[1, 0, 0]], [[0, 0, 0]], timestamp=stamp, now=100.)
        points = [[.02, .05, .05], [.08, .05, .05], [.15, .05, .05],
                  [.25, .05, .05], [.4, .05, .05], [np.nan, .05, .05]]
        result = score_surface_events(occupancy.snapshot(), points, ['a', 'a', 'a', 'b', 'b', 'a'],
                                      camera_ids=['A', 'B', 'A', 'B', 'A', 'B'])
        self.assertEqual((result['input_events'], result['finite_events'], result['invalid_events']), (6, 5, 1))
        self.assertEqual(result['per_label']['a']['events'], 3)
        self.assertEqual(result['per_label']['a']['event_states']['occupied'], 2)
        self.assertEqual(result['per_label']['a']['unique_in_bounds_voxels'], 2)
        self.assertEqual(result['overall']['unique_voxel_states']['occupied'], 1)
        self.assertEqual(result['overall']['event_states']['out_of_bounds'], 1)

    def test_clear_gap_can_have_no_wholly_clear_cell_and_invalid_samples_are_retained(self):
        spec = GridSpec((0., 0., 0.), (.03, .01, .01), .01)
        occupancy = OccupancyMap(spec, clock=lambda: 100.)
        occupancy.integrate([], [[1, 0, 0]], timestamp=99., now=100.)
        shapes = [box('left', (.003, .005, .005), (.006, .01, .01)),
                  box('right', (.023, .005, .005), (.014, .01, .01))]
        result = score_clear_region(occupancy.snapshot(), [[.012, .005, .005]], shapes)
        self.assertTrue(result['valid'])
        self.assertEqual(result['map_states']['occupied'], 1)
        self.assertEqual(result['conservative_whole_cell_clear']['voxels'], 0)
        self.assertEqual(result['possible_cell_overlap']['voxels'], 1)
        invalid = score_clear_region(occupancy.snapshot(), [[.003, .005, .005], [.05, .005, .005]], shapes)
        self.assertFalse(invalid['valid'])
        self.assertEqual(set(invalid['errors']), {'probe_intersects_reference_solid', 'out_of_bounds_probe_points'})
        self.assertEqual(invalid['solid_points'], 1)
        self.assertEqual(invalid['input_points'], 2)

    def test_empty_domains_have_null_rates_and_expired_history_is_not_current_observation(self):
        spec = GridSpec((0., 0., 0.), (.1, .1, .1), .1)
        occupancy = OccupancyMap(spec, clock=lambda: 100., evidence_ttl=.5)
        occupancy.integrate([], [[0, 0, 0]], timestamp=99., now=99.)
        snapshot = occupancy.snapshot(now=100.)
        empty = score_voxel_centers(snapshot, [], [])
        self.assertIsNone(empty['unknown']['rate'])
        scored = score_voxel_centers(snapshot, [], [[0, 0, 0]])
        self.assertEqual(scored['observed_region']['voxels'], 0)
        self.assertEqual(scored['reference_free']['unknown'], 1)
        self.assertFalse(score_clear_region(snapshot, [], [])['valid'])
        json.dumps(scored, allow_nan=False)

    def test_incomplete_reference_exclusions_are_explicit_denominator_omissions(self):
        spec = GridSpec((0., 0., 0.), (.3, .1, .1), .1)
        occupancy = OccupancyMap(spec, clock=lambda: 100.)
        occupancy.integrate([], [[0, 0, 0], [1, 0, 0]], timestamp=99., now=100.)
        indices = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        result = score_voxel_centers(occupancy.snapshot(), [], indices, excluded_indices=[[0, 0, 0]])
        self.assertEqual((result['input_voxels'], result['excluded_voxels'], result['evaluated_voxels']), (3, 1, 2))
        self.assertEqual(result['false_occupied'], dict(count=1, denominator=2, rate=.5))
        region = score_clear_region(occupancy.snapshot(), [[.05, .05, .05]], [], excluded_indices=[[0, 0, 0]])
        self.assertFalse(region['valid'])
        self.assertEqual(region['errors'], ['probe_in_excluded_reference_region'])
        self.assertEqual(region['conservative_whole_cell_clear']['voxels'], 0)


if __name__ == '__main__':
    unittest.main()

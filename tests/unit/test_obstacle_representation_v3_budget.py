"""Budget diagnostics preserve nested original geometry and honest norm claims."""
import unittest

from scripts.probe_obstacle_representation_v3_budget import nested_selections, nondecreasing, norm


class ObstacleRepresentationV3BudgetTests(unittest.TestCase):
    def test_nested_packet_check_preserves_numeric_geometry_and_original_expiry(self):
        snapshot = dict(voxel_size=.01, bounds_min=[0., 0., 0.])
        boxes = [dict(id=str(i), center=[(i+.5)*.01, .005, .005], dimensions=[.01]*3,
            expires_at=10., voxel_count=1) for i in range(4)]
        packets = {name: dict(boxes=boxes[:count]) for name, count in
            [('b16', 1), ('b32', 2), ('baseline', 3), ('b128', 4)]}
        checks = nested_selections(packets, snapshot)
        self.assertTrue(all(value['selected_cells_subset'] and value['numeric_cuboid_prefix'] for value in checks))
        packets['b128']['boxes'] = [dict(box, expires_at=11.) for box in boxes]
        with self.assertRaises(ValueError):
            nested_selections(packets, snapshot)

    def test_force_vector_cancellation_can_reduce_norm_after_adding_a_term(self):
        first, last = [2., 0.], [1., 0.]
        self.assertFalse(nondecreasing([norm(first), norm(last)]))
        self.assertTrue(nondecreasing([1., 1.-1e-12]))

    def test_matrix_norm_is_frobenius(self):
        self.assertEqual(norm([[3., 0.], [0., 4.]]), 5.)


if __name__ == '__main__':
    unittest.main()

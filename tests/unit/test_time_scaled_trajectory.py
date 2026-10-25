"""Exact geometry, derivatives and domain bounds of constant time scaling."""
import unittest

import numpy as np

from robot_skill_stack.manipulation.motion.trajectory import TimeScaledTrajectory


class PolynomialTrajectory:
    start_time, end_time = 2.5, 3.3

    def __init__(self):
        self.trajectory = self

    def get_active_joints(self):
        return [f'panda_joint{i}' for i in range(1, 8)]

    def eval(self, time, derivative):
        offset = time-self.start_time
        value = (1+2*offset+3*offset**2+4*offset**3,
                 2+6*offset+12*offset**2, 6+24*offset, 24)[derivative]
        return np.arange(1, 8)*value

    def get_joint_targets(self, time):
        return self.eval(time, 0), self.eval(time, 1)


class TimeScaledTrajectoryTests(unittest.TestCase):
    def test_endpoints_and_nonzero_start_time_are_exact(self):
        base = PolynomialTrajectory()
        scaled = TimeScaledTrajectory(base, 2.)
        self.assertEqual(scaled.start_time, 2.5)
        self.assertEqual(scaled.end_time, 4.1)
        for first, last in ((base.start_time, scaled.start_time), (base.end_time, scaled.end_time)):
            np.testing.assert_array_equal(scaled.get_joint_targets(last)[0], base.get_joint_targets(first)[0])
        self.assertEqual(scaled.get_active_joints(), base.get_active_joints())

    def test_curve_identity_and_all_derivative_chain_rule_factors(self):
        base = PolynomialTrajectory()
        scaled = TimeScaledTrajectory(base, 2.)
        for stamp in np.linspace(base.start_time, base.end_time, 17):
            mapped = scaled.start_time+2*(stamp-base.start_time)
            q, qd = scaled.get_joint_targets(mapped)
            np.testing.assert_allclose(q, base.eval(stamp, 0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(qd, base.eval(stamp, 1)/2, rtol=0, atol=1e-12)
            for derivative in range(4):
                np.testing.assert_allclose(scaled.trajectory.eval(mapped, derivative),
                    base.eval(stamp, derivative)/2**derivative, rtol=0, atol=1e-12)

    def test_invalid_scale_and_out_of_domain_queries_are_rejected(self):
        for factor in (0., .5, 4.1, float('inf'), float('nan'), True):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                TimeScaledTrajectory(PolynomialTrajectory(), factor)
        scaled = TimeScaledTrajectory(PolynomialTrajectory())
        for stamp in (2.49, 4.11, float('nan'), float('inf')):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                scaled.get_joint_targets(stamp)

    def test_nested_scaling_is_bounded_and_preserves_composition(self):
        base = PolynomialTrajectory()
        nested = TimeScaledTrajectory(TimeScaledTrajectory(base, 2), 2)
        self.assertEqual(nested.time_scale_factor, 4.)
        np.testing.assert_allclose(nested.eval(nested.end_time, 3), base.eval(base.end_time, 3)/64)
        with self.assertRaises(ValueError):
            TimeScaledTrajectory(nested, 2)


if __name__ == '__main__':
    unittest.main()

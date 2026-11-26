"""Shared numeric conventions and immutable-copy ownership."""
import unittest

import numpy as np

from robot_skill_stack.common.arrays import readonly_copy
from robot_skill_stack.common.rotations import quat_matrix


class CommonTests(unittest.TestCase):
    def test_readonly_copy_preserves_shape_dtype_values_and_c_order(self):
        values = (np.arange(12, dtype=np.uint8).reshape(3, 4),
                  np.arange(24, dtype='>f8').reshape(4, 6)[::-1, ::2],
                  np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
                  np.empty((0, 3), dtype=np.int64), np.array(4, dtype=np.int16))
        for source in values:
            with self.subTest(shape=source.shape, dtype=source.dtype):
                result = readonly_copy(source)
                self.assertEqual((result.shape, result.dtype), (source.shape, source.dtype))
                np.testing.assert_array_equal(result, source)
                self.assertTrue(result.flags.c_contiguous)
                self.assertFalse(result.flags.writeable)
                with self.assertRaises(ValueError):
                    result.setflags(write=True)
                base = result
                while isinstance(base, np.ndarray):
                    base = base.base
                self.assertIsInstance(base, bytes)

    def test_readonly_copy_always_has_independent_storage(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = readonly_copy(source)
        source[:] = -1
        np.testing.assert_array_equal(result, np.arange(12).reshape(3, 4))
        copied_again = readonly_copy(result)
        self.assertFalse(np.shares_memory(result, copied_again))
        np.testing.assert_array_equal(copied_again, result)

    def test_readonly_copy_converts_only_when_dtype_is_requested(self):
        source = [[1, 2], [3, 4]]
        result = readonly_copy(source, dtype='>i2')
        self.assertEqual(result.dtype, np.dtype('>i2'))
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_array_equal(result, source)

    def test_quaternion_conversion_normalizes_without_lulas_minimum_norm_policy(self):
        for norm in (1., 2.5, 1e-13):
            np.testing.assert_array_equal(quat_matrix([norm, 0, 0, 0]), np.eye(3))
        np.testing.assert_array_equal(quat_matrix([0, 0, 0, 2]), np.diag([-1, -1, 1]))
        with self.assertRaises(ValueError):
            quat_matrix([0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

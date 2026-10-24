"""Prepared cuboids use SDK boundaries without creating simulator geometry."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.obstacles import RmpflowObstacleRegistry
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleBox, ObstaclePacket


def box(key='A', center=(.4, 0., .3), dimensions=(.1, .1, .1), expires=100.):
    return ObstacleBox(key, center, dimensions, expires, 8)


def packet(*boxes, revision=1, generation=0, timestamp=10.):
    return ObstaclePacket(generation, revision, timestamp, timestamp+.1, boxes, {})


class Policy:
    def __init__(self, units=1.):
        self.units = units
        self.registered = {}
        self.calls = []
        self.fail_after_add = self.fail_remove_once = False

    def add_cuboid(self, obstacle, static=False):
        assert static is False
        self.calls.append(('add', obstacle))
        self.registered[obstacle] = dict(
            center=obstacle.get_world_pose()[0]*self.units,
            dimensions=obstacle.get_size()*obstacle.get_local_scale()*self.units)
        if self.fail_after_add:
            self.fail_after_add = False
            raise RuntimeError('SDK failed after registering cuboid')
        return True

    def remove_obstacle(self, obstacle):
        self.calls.append(('remove', obstacle))
        if self.fail_remove_once:
            self.fail_remove_once = False
            raise RuntimeError('SDK remove failed once')
        return self.registered.pop(obstacle, None) is not None

    def update_world(self, updated_obstacles=None):
        self.calls.append(('world', updated_obstacles))
        for obstacle, record in self.registered.items():
            if hasattr(obstacle, 'get_world_pose'):
                record['center'] = obstacle.get_world_pose()[0]*self.units


class RmpflowObstacleRegistryTests(unittest.TestCase):
    def setUp(self):
        self.policy = Policy()
        self.registry = RmpflowObstacleRegistry(self.policy, meters_per_unit=1.)

    def test_numeric_getters_preserve_meters_without_sharing_mutable_arrays(self):
        policy = Policy(units=.01)
        registry = RmpflowObstacleRegistry(policy, meters_per_unit=.01)
        registry.sync(packet(box()), now=11.)
        proxy, record = next(iter(policy.registered.items()))
        np.testing.assert_array_equal(record['center'], [.4, 0., .3])
        np.testing.assert_array_equal(record['dimensions'], [.1, .1, .1])
        position, orientation = proxy.get_world_pose()
        position[:] = 99.
        orientation[:] = 99.
        scale = proxy.get_local_scale()
        scale[:] = 99.
        np.testing.assert_array_equal(proxy.get_world_pose()[0], [40., 0., 30.])
        np.testing.assert_array_equal(proxy.get_world_pose()[1], [1., 0., 0., 0.])
        np.testing.assert_array_equal(proxy.get_local_scale(), [10., 10., 10.])

    def test_unchanged_packet_and_new_revision_without_geometry_change_make_no_sdk_calls(self):
        first = packet(box())
        self.assertTrue(self.registry.sync(first, now=11.))
        before = list(self.policy.calls)
        self.assertFalse(self.registry.sync(first, now=12.))
        self.assertFalse(self.registry.sync(packet(box(), revision=2, timestamp=11.), now=12.))
        self.assertEqual(self.policy.calls, before)
        self.assertEqual(self.registry.diagnostics['revision'], 2)
        self.assertEqual(self.registry.diagnostics['timestamp'], 11.)

    def test_pose_changes_reuse_registered_proxy_and_dimensions_readd_only_that_proxy(self):
        self.registry.sync(packet(box(), box('B', center=(.8, 0., .3))), now=11.)
        first = dict(self.registry._proxies)
        self.registry.sync(packet(box(center=(.41, 0., .3)), box('B', center=(.8, 0., .3)), revision=2), now=11.)
        self.assertIs(self.registry._proxies['A'], first['A'])
        self.assertEqual(self.registry.diagnostics['additions'], 2)
        np.testing.assert_array_equal(self.policy.registered[first['A']]['center'], [.41, 0., .3])
        self.registry.sync(packet(box(dimensions=(.12, .1, .1)), box('B', center=(.8, 0., .3)), revision=3), now=11.)
        self.assertIs(self.registry._proxies['A'], first['A'])
        self.assertIs(self.registry._proxies['B'], first['B'])
        self.assertEqual(self.registry.diagnostics['dimension_updates'], 1)
        self.assertEqual(self.registry.diagnostics['additions'], 3)
        self.assertEqual(self.registry.diagnostics['removals'], 1)

    def test_moving_grid_ids_reuse_registration_without_joining_disjoint_boxes(self):
        self.registry.sync(packet(box('oldA'), box('oldB', center=(.8, 0., .3))), now=11.)
        first = dict(self.registry._proxies)
        self.registry.sync(packet(box('newA', center=(.41, 0., .3)),
            box('newB', center=(.81, 0., .3)), revision=2), now=11.)
        self.assertIs(self.registry._proxies['newA'], first['oldA'])
        self.assertIs(self.registry._proxies['newB'], first['oldB'])
        self.assertEqual(self.registry.diagnostics['additions'], 2)
        self.assertEqual(self.registry.diagnostics['identity_reuses'], 2)
        self.assertEqual(len(self.policy.registered), 2)
        np.testing.assert_array_equal(self.policy.registered[first['oldA']]['dimensions'], [.1, .1, .1])

    def test_far_new_geometry_removes_old_registration(self):
        self.registry.sync(packet(box()), now=11.)
        old = self.registry._proxies['A']
        self.registry.sync(packet(box('C', center=(1.2, 0., .3)), revision=2), now=11.)
        self.assertNotIn(old, self.policy.registered)
        self.assertEqual(self.registry.diagnostics['proxy_count'], 1)

    def test_expiry_removes_old_evidence_even_without_revision_change(self):
        source = packet(box(expires=12.), box('B', center=(.8, 0., .3), expires=15.))
        self.registry.sync(source, now=11.)
        self.registry.sync(source, now=12.)
        self.assertEqual(set(self.registry._proxies), {'B'})
        before = list(self.policy.calls)
        self.registry.sync(source, now=13.)
        self.assertEqual(self.policy.calls, before)
        self.registry.sync(source, now=15.)
        self.assertEqual(self.registry.diagnostics['registered'], 0)

    def test_generation_change_rebuilds_only_owned_obstacles_and_old_generation_is_rejected(self):
        foreign = object()
        self.policy.registered[foreign] = {}
        self.registry.sync(packet(box()), now=11.)
        old = self.registry._proxies['A']
        self.registry.sync(packet(box(), generation=1), now=11.)
        self.assertIsNot(self.registry._proxies['A'], old)
        self.assertIn(foreign, self.policy.registered)
        before = list(self.policy.calls)
        with self.assertRaisesRegex(ValueError, 'Obsolete'):
            self.registry.sync(packet(box(), generation=0), now=11.)
        self.assertEqual(self.policy.calls, before)

    def test_old_revision_timestamp_and_conflicting_revision_cannot_refresh_obstacles(self):
        self.registry.sync(packet(box(), revision=3), now=11.)
        before = list(self.policy.calls)
        invalid = (packet(box(), revision=2), packet(box(), revision=4, timestamp=9.),
            packet(box(center=(.5, 0., .3)), revision=3), packet(box(), revision=3, timestamp=10.5))
        for source in invalid:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'Obsolete|Conflicting'):
                self.registry.sync(source, now=11.)
            self.assertEqual(self.policy.calls, before)

    def test_bad_or_oversized_packet_is_rejected_before_mutating_existing_sdk_state(self):
        self.registry.sync(packet(box()), now=11.)
        before = list(self.policy.calls)
        def untrusted_box(**values):
            return NS(**dict(dict(id='A', center=(.4, 0., .3),
                dimensions=(.1, .1, .1), expires_at=100.), **values))
        bad = (packet(box(), box()), packet(untrusted_box(center=(np.nan, 0., 0.))),
            packet(untrusted_box(dimensions=(0., .1, .1))), packet(untrusted_box(expires_at=np.nan)),
            packet(box(), timestamp=12.), packet(box(), generation=-1),
            packet(*(box(str(i)) for i in range(65))))
        for source in bad:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.registry.sync(source, now=11.)
            self.assertEqual(self.policy.calls, before)

    def test_clear_and_idempotent_close_leave_other_policies_registrations_alone(self):
        foreign = object()
        self.policy.registered[foreign] = {}
        self.registry.sync(packet(box()), now=11.)
        self.registry.close()
        self.registry.close()
        self.assertEqual(set(self.policy.registered), {foreign})
        self.assertTrue(self.registry.diagnostics['closed'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.registry.sync(packet(box()), now=11.)

    def test_reset_tracking_handles_sdk_controller_reset_without_reusing_erased_handles(self):
        source = packet(box())
        self.registry.sync(source, now=11.)
        first = self.registry._proxies['A']
        self.policy.registered.clear()  # The backend's existing SDK reset.
        self.registry.reset_tracking()
        before = len(self.policy.calls)
        self.registry.sync(source, now=11.)
        self.assertIsNot(self.registry._proxies['A'], first)
        self.assertFalse(any(name == 'remove' for name, _ in self.policy.calls[before:]))
        self.assertEqual(len(self.policy.registered), 1)

    def test_owner_thread_guard_precedes_every_sdk_call(self):
        source = packet(box())
        with ThreadPoolExecutor(max_workers=1) as executor:
            for call in (lambda: self.registry.sync(source, now=11.), self.registry.clear,
                         self.registry.close, self.registry.reset_tracking):
                with self.assertRaisesRegex(RuntimeError, 'owning thread'):
                    executor.submit(call).result(timeout=1.)
        self.assertEqual(self.policy.calls, [])

    def test_failed_sdk_registration_remains_owned_until_cleanup_can_succeed(self):
        self.policy.fail_after_add = self.policy.fail_remove_once = True
        with self.assertRaisesRegex(RuntimeError, 'after registering'):
            self.registry.sync(packet(box()), now=11.)
        self.assertEqual(self.registry.diagnostics['registered'], 1)
        self.registry.clear()
        self.assertEqual(self.policy.registered, {})


if __name__ == '__main__':
    unittest.main()

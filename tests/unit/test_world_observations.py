"""Source contract and updater behavior with a controlled monotonic clock."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from robot_skill_stack.common.types import Pose
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel


def source():
    return NS(observe=Mock(return_value=ObservationBatch()), forget=Mock(), close=Mock())


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.model, self.source = WorldModel(), source()
        self.updater = WorldModelUpdater(self.model, self.source, stale_object_ttl=2.)

    def frame(self, observations):
        self.now.return_value += .2
        self.source.observe.return_value = ObservationBatch(observations, self.now.return_value)
        return self.updater.update()

    def seen(self):
        self.frame([ObjectObservation('cube', pose=Pose([.45, 0, .025]))])
        return self.model.require('cube')

    def test_fresh_empty_and_no_frame_are_distinct(self):
        self.assertEqual(self.updater.health.status, 'waiting')
        obj = self.seen()
        self.assertEqual(self.updater.health.status, 'fresh')
        seen = obj.last_seen
        self.source.observe.return_value = ObservationBatch()
        self.updater.update()
        self.assertTrue(obj.visible)
        self.assertEqual(obj.last_seen, seen)
        self.frame([])
        self.assertFalse(obj.visible)
        self.assertEqual(self.updater.health.status, 'fresh')
        self.assertEqual(obj.last_seen, seen)

    def test_frozen_or_repeated_sample_never_refreshes_cached_state(self):
        obj = self.seen()
        original = obj.last_seen
        self.now.return_value += 1.1
        self.updater.update()  # Source incorrectly repeats its old batch.
        self.assertEqual(self.updater.health.status, 'stale')
        self.assertFalse(obj.visible)
        self.assertEqual(obj.last_seen, original)
        self.now.return_value += 1.
        self.updater.update()
        self.assertFalse(self.model.exists('cube'))
        self.source.forget.assert_called_once_with('cube')
        self.assertEqual(self.updater.update_count, 1)

    def test_failure_is_not_empty_and_requires_a_fresh_sample_to_recover(self):
        obj = self.seen()
        self.source.observe.side_effect = RuntimeError('disconnected')
        self.updater.update()
        self.assertFalse(obj.visible)
        self.assertEqual(self.updater.health.status, 'failed')
        self.assertIn('disconnected', self.updater.last_error)
        self.source.observe.side_effect = None
        self.source.observe.return_value = ObservationBatch()
        self.updater.update()
        self.assertEqual(self.updater.health.status, 'failed')
        self.frame([])
        self.assertEqual(self.updater.health.status, 'fresh')
        self.assertIsNone(self.updater.last_error)
        self.assertFalse(obj.visible)

    def test_same_id_reacquires_before_ttl_and_hint_consumes_only_on_match(self):
        obj = self.seen()
        self.frame([])
        self.model.expect_object_at('cube', [.5, .2, .025])
        self.frame([ObjectObservation('cube', pose=Pose([.5, .2, .025]),
                                      metadata={'association_hint_match': True})])
        self.assertIs(self.model.require('cube'), obj)
        self.assertTrue(obj.visible)
        self.assertFalse(self.model.association_hints())

    def test_held_uncertain_and_place_hint_protect_identity_not_visibility(self):
        for protection in ('held', 'uncertain', 'hint'):
            with self.subTest(protection=protection):
                obj = self.seen()
                if protection == 'hint':
                    self.model.expect_object_at('cube', [.5, .2, .025], ttl=10.)
                else:
                    self.model.set_held('cube', object_offset_in_ee=[0, 0, .03],
                                        uncertain=protection == 'uncertain')
                self.now.return_value += 3.
                self.updater.cleanup()
                self.assertIs(self.model.require('cube'), obj)
                self.assertFalse(obj.visible)
                self.assertEqual(self.updater.clear_lost(), ())
                self.model.clear_held_state()
                self.model.clear_association_hint('cube')
                self.assertEqual(self.updater.cleanup(), ('cube',))

    def test_one_expiration_policy_includes_ground_truth(self):
        obj = self.seen()
        obj.source = 'ground_truth'
        self.now.return_value += 3.
        self.assertEqual(self.updater.cleanup(), ('cube',))
        self.source.forget.assert_called_once_with('cube')

    def test_hint_reads_are_pure_and_updater_discards_expired_hints(self):
        self.seen()
        self.model.expect_object_at('cube', [.5, .2, .025], ttl=.1)
        self.now.return_value += .2
        before = dict(self.model._association_hints)
        self.assertEqual(self.model.association_hints(), ())
        self.assertEqual(self.model._association_hints, before)
        self.updater.cleanup()
        self.assertFalse(self.model._association_hints)

    def test_cleanup_between_scheduled_samples(self):
        obj = self.seen()
        self.updater._period = 5.
        self.now.return_value += 1.1
        self.updater.tick(.1)
        self.assertFalse(obj.visible)
        self.assertEqual(self.updater.update_count, 1)

    def test_timestamp_in_wrong_clock_domain_fails_closed(self):
        self.source.observe.return_value = ObservationBatch([], float('nan'))
        self.updater.update()
        self.assertEqual(self.updater.health.status, 'failed')
        self.assertIn('monotonic', self.updater.last_error)

    def test_stale_source_blocks_new_pick_and_place_without_robot_commands(self):
        from tests.unit.test_held_state import HeldStateTests
        # Use the established manipulation fixture; no control emulation here.
        model, backend, runtime = HeldStateTests().stack()
        updater = WorldModelUpdater(model, source())
        for held in (False, True):
            if held:
                model.set_held('target', object_offset_in_ee=[0, 0, .03])
            result = asyncio.run(runtime.execute_async(
                'place' if held else 'pick',
                **({'target': Pose([.45, .25, .05])} if held else {'object_id': 'target'})))
            self.assertEqual(result.failure_code, FailureCode.WORLD_STATE_UNAVAILABLE)
            self.assertEqual(backend.calls, [])

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from scripts import validate_global_stabilization_v4_1 as validation


class ProductionValidationTests(unittest.TestCase):
    def test_hold_observation_follows_original_callback_and_preserves_its_result(self):
        calls, observer = [], object()
        outcome = dict(status='PASS', excluded_from_primary_timing=True)

        def original_factory(mode, state):
            self.assertEqual(mode, 'C')
            async def original(*args):
                calls.append('original-complete')
                state['callback_complete'] = True
                return outcome
            return original

        async def hold(current, seconds):
            self.assertIs(current, observer)
            self.assertEqual(seconds, 1.)
            calls.append('observe-hold')

        state = dict(observer=observer)
        with patch.object(validation.observations, 'observe_post_home_hold', AsyncMock(side_effect=hold)):
            result = asyncio.run(validation.callback_for('C', state, original_factory)())
        self.assertIs(result, outcome)
        self.assertEqual(calls, ['original-complete', 'observe-hold'])
        self.assertTrue(state['callback_complete'])

    def test_duration_defaults_to_one_and_five_is_forwarded_only_to_hold_observer(self):
        for options, expected in [([], 1.), (['--post-hold-seconds', '5'], 5.),
                (['--post-hold-seconds=5'], 5.)]:
            with self.subTest(options=options):
                observer, outcome = object(), dict(status='PASS')
                state = dict(observer=observer)
                def original_factory(mode, current):
                    self.assertEqual(mode, 'C')
                    self.assertIs(current, state)
                    return AsyncMock(return_value=outcome)
                def controlled_main(arguments):
                    self.assertEqual(arguments, ['--mode', 'C'])
                    self.assertIs(asyncio.run(validation.controlled.callback_for('C', state)()), outcome)
                    return True
                with patch.object(validation.controlled, 'callback_for', original_factory), \
                        patch.object(validation.controlled, 'main', side_effect=controlled_main), \
                        patch.object(validation.observations, 'observe_post_home_hold', AsyncMock()) as hold:
                    self.assertTrue(validation.main(['--mode', 'C', *options]))
                hold.assert_awaited_once_with(observer, expected)


if __name__ == '__main__':
    unittest.main()

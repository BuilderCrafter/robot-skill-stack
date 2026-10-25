"""Control finalization cannot revise the locked primary experiment."""
import copy
import unittest

import numpy as np

from scripts.finalize_global_detour_v4_3 import assert_locked_main,override_goal_control,select_control


class DetourFinalizationTests(unittest.TestCase):
    def test_changing_primary_geometry_or_obstruction_is_rejected(self):
        selected=dict(center=np.array([.55,-.225,.3]),dimensions=np.array([.06,.06,.12]),
            planned=dict(radius=.0031),actual=dict(radius=.0037))
        lock=dict(locked_main_fixture=dict(position_m=[.55,-.225,.3],dimensions_m=[.06,.06,.12],
            orientation_wxyz=[1.,0.,0.,0.]),planned_obstruction=selected['planned'],actual_obstruction=selected['actual'])
        assert_locked_main(lock,selected)
        changed=copy.deepcopy(selected);changed['center'][2]+=.001
        with self.assertRaisesRegex(ValueError,'locked main geometry'):assert_locked_main(lock,changed)
        changed=copy.deepcopy(selected);changed['planned']['radius']+=.001
        with self.assertRaisesRegex(ValueError,'obstruction evidence'):assert_locked_main(lock,changed)

    def test_only_goal_control_dimensions_are_overridden(self):
        original=dict(grid={'fixed':True},threshold=.01,negative_controls=dict(near=dict(size=[.09,.14,.25]),
            goal_blocked=dict(dimensions_m=[.18]*3,position='fixed goal')))
        result=override_goal_control(original,.06)
        self.assertEqual(result['negative_controls']['goal_blocked']['dimensions_m'],[.06]*3)
        self.assertEqual(original['negative_controls']['goal_blocked']['dimensions_m'],[.18]*3)
        self.assertEqual(result['grid'],original['grid'])
        self.assertEqual(result['negative_controls']['near'],original['negative_controls']['near'])

    def test_largest_first_control_requires_both_physical_and_native_checks(self):
        protocol=dict(goal_blocked_control=dict(candidate_cube_edge_lengths_m=[.08,.06,.04]),
            unchanged_criteria=dict(minimum_actual_home_clearance_m=.01,minimum_goal_intersection_ball_radius_m=.003))
        def verify(side):return dict(actual_home_clearance_m=.02,enhanced_home_clear=side<.08,
            enhanced_goal_blocked=True,goal_intersection_radius_m=.005)
        side,records=select_control(protocol,verify)
        self.assertEqual(side,.06)
        self.assertEqual([row['cube_edge_length_m'] for row in records],[.08,.06])
        def invalid(side):return dict(actual_home_clearance_m=.02,enhanced_home_clear=True,
            enhanced_goal_blocked=False,goal_intersection_radius_m=.005)
        side,records=select_control(protocol,invalid)
        self.assertIsNone(side)
        self.assertEqual(len(records),3)


if __name__=='__main__':unittest.main()

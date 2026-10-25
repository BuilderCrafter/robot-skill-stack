"""Exact convex fallback preserves the declared margins and bounded selection."""
import json
import unittest

import numpy as np

from scripts import design_global_detour_v4_3 as original
from scripts.design_global_detour_exact_v4_3 import PROTOCOL,certify_distances,fast_intersection


class ExactGlobalDetourDesignTests(unittest.TestCase):
    def test_revision_keeps_original_grid_library_and_accepted_thresholds(self):
        before=json.loads(original.PROTOCOL.read_text())
        after=json.loads(PROTOCOL.read_text())
        for field in ('candidate_grid_integer_steps','candidate_dimensions_m','candidate_orientation_wxyz',
                'witness_priority','maximum_joint_interval_rad','minimum_endpoint_cooked_clearance_m',
                'minimum_original_route_intersection_ball_radius_m',
                'minimum_continuous_witness_external_lower_bound_m','detour_acceptance'):
            self.assertEqual(before[field],after[field],field)
        self.assertEqual(after['candidate_RRT_calls_allowed_during_design'],0)

    def test_insufficient_aabb_is_not_treated_as_physical_collision(self):
        # Two narrow diagonal solids have overlapping world AABBs, yet a small
        # box near the opposite corner is separated from the actual solid.
        first=original.box_vertices([0.,0.,0.],[1.,.02,.02])
        angle=np.pi/4
        rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        vertices=first@rotation.T
        center=np.array([.25,-.25,0.]);size=np.array([.04,.04,.04])
        self.assertEqual(original.aabb_distance(vertices.min(0),vertices.max(0),center,size/2),0.)
        result=certify_distances([[vertices]],[dict(body='finger')],center,size,np.array([.01]))
        self.assertTrue(result['passed'])
        self.assertEqual(result['convex_queries'],1)
        self.assertGreater(result['minimum_lower_bound_m'],.3)

    def test_failed_exact_margin_does_not_become_an_avoidance_success(self):
        vertices=original.box_vertices([0.,0.,0.],[.2,.2,.2])
        result=certify_distances([[vertices]],[dict(body='hand')],np.array([.2,0.,0.]),np.array([.2]*3),np.array([.01]))
        self.assertFalse(result['passed'])
        self.assertLess(result['lower_bound_m'],.01)
        self.assertIn('not itself a collision proof',result['interpretation'])

    def test_vectorized_obstruction_uses_same_positive_volume_criterion(self):
        solid=original.box_vertices([0.,0.,0.],[1.,1.,1.])
        world=[[solid],[solid+np.array([2.,0.,0.])]]
        lower=np.array([[row.min(0) for row in state] for state in world])
        upper=np.array([[row.max(0) for row in state] for state in world])
        proof=fast_intersection(world,[dict(body='panda_hand')],lower,upper,np.array([2.8,0.,0.]),np.ones(3),.01,times=[0.,.3])
        self.assertEqual(proof['baseline_sample'],1)
        self.assertEqual(proof['planned_time'],.3)
        self.assertGreater(proof['positive_ball_radius_lower_bound_m'],.09)


if __name__=='__main__':unittest.main()

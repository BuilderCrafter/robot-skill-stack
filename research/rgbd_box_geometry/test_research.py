"""Portable focused checks; archived replay is opt-in through an external path."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

from . import fixed_grid,reference as ref,visibility
from .inputs import PACKAGE,load_capture


def samples(state='supported'):
    return [dict(grid_bin=[i,j],pixel_xy=[i,j],state=state) for i in range(3) for j in range(3)]


class VisibilityTests(unittest.TestCase):
    def test_fov_does_not_supply_positive_evidence(self):
        s=samples()
        for p in s:
            if p['grid_bin'][1]==2:p['state']='outside_fov'
        decision=visibility.patch_decision(s)
        self.assertTrue(decision['qualified'])
        self.assertEqual(decision['unique_supported'],6)
        self.assertEqual(decision['state_counts']['outside_fov'],3)

    def test_wholly_unobservable_patch_fails(self):
        for state in visibility.POLICY['unobserved']:
            with self.subTest(state=state):
                self.assertFalse(visibility.patch_decision(samples(state))['qualified'])

    def test_partial_visibility_alone_is_not_enough(self):
        s=samples('outside_fov')
        for p in s[:3]:p['state']='supported'
        self.assertFalse(visibility.patch_decision(s)['qualified'])

    def test_contradiction_cannot_be_hidden_by_other_support(self):
        s=samples();s[-1]['state']='contradictory_free'
        self.assertFalse(visibility.patch_decision(s)['qualified'])

    def test_duplicate_pixel_samples_are_not_independent_evidence(self):
        s=samples()
        for p in s:p['pixel_xy']=[1,1]
        self.assertFalse(visibility.patch_decision(s)['qualified'])

    def test_depth_states(self):
        self.assertEqual(visibility.classify(False,1.,1.),'outside_fov')
        self.assertEqual(visibility.classify(True,-1.,1.),'behind_camera')
        self.assertEqual(visibility.classify(True,1.,np.nan),'invalid_depth')
        self.assertEqual(visibility.classify(True,1.,.9),'occluded')
        self.assertEqual(visibility.classify(True,1.,1.1),'contradictory_free')
        self.assertEqual(visibility.classify(True,1.,1.),'supported')

    def test_missing_bound_and_robot_guards_remain_active(self):
        old=json.loads((PACKAGE/'evidence/reference/blind-results.json').read_text())
        c=next(c for c in old['v43']['components'] if c.get('hypothesis'))
        c['faces']['z-1']['qualified']=False
        result=visibility.qualify(c,())
        self.assertEqual(result['status'],'unqualified')
        self.assertIsNone(result['geometry'])
        c['robot_envelope_points']=1
        self.assertEqual(visibility.qualify(c,())['reason'],c['reason'])

    def test_preserved_reference_and_evidence_hashes(self):
        lineage=json.loads((PACKAGE/'lineage.json').read_text())
        self.assertEqual(ref.digest(PACKAGE/'reference.py'),lineage['reference_original_sha256'])
        self.assertEqual(ref.digest(PACKAGE/'fixed_grid.py'),lineage['fixed_grid_published_sha256'])
        for name,expected in lineage['historical_evidence'].items():
            self.assertEqual(ref.digest(PACKAGE/name),expected)


@unittest.skipUnless(os.environ.get('RGBD_RESEARCH_CAPTURE_ROOT'),'Set RGBD_RESEARCH_CAPTURE_ROOT to trusted external archives')
class ArchivedReplayTests(unittest.TestCase):
    def test_blind_reference_determinism_and_camera_provenance(self):
        expected=json.loads((PACKAGE/'evidence/reference/blind-results.json').read_text())
        for case in ('static','v43'):
            with self.subTest(case=case):
                payload,scene,_=load_capture(Path(os.environ['RGBD_RESEARCH_CAPTURE_ROOT']),case)
                result,_,_=ref.reconstruct(payload.frames,scene,payload.robot_snapshots)
                result.pop('timing_ms')
                self.assertEqual(ref.plain(result),expected[case])
                self.assertEqual(result['source_timestamp'],min(f.acquired_at for f in payload.frames))
                snapshots=dict(payload.robot_snapshots)
                f=payload.frames[0]
                snapshots[f.camera_id]=replace(snapshots[f.camera_id],frame_token=(f.token[0]+1,f.token[1]))
                with self.assertRaises(ValueError):ref.robot_envelopes(payload.frames,snapshots)


if __name__=='__main__':unittest.main()

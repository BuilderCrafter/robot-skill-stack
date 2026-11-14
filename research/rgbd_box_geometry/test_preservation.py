"""Post-freeze validation of published artifacts; never changes qualification."""
import ast
import json
import os
from pathlib import Path
import unittest

import numpy as np

from . import controls,fixed_grid,reference as ref,visibility
from .inputs import PACKAGE,load_capture
from .run import verify_seal


class PreservationTests(unittest.TestCase):
    def test_all_published_blind_seals(self):
        for name in ('reference','fixed_grid','visibility'):
            with self.subTest(name=name):verify_seal(PACKAGE/'evidence'/name)

    def test_runtime_sources_do_not_import_research(self):
        # Publication must not create a production dependency on these experiments.
        root=PACKAGE.parents[1]/'robot_skill_stack'
        if not root.exists():self.skipTest('Application checkout not adjacent to package')
        for path in root.rglob('*.py'):
            for n in ast.walk(ast.parse(path.read_text())):
                names=[]
                if isinstance(n,ast.Import):names=[a.name for a in n.names]
                elif isinstance(n,ast.ImportFrom):names=[n.module or '']
                self.assertFalse(any(x=='research' or x.startswith('research.') for x in names),str(path))

    def test_evidence_and_declared_sources_are_preserved(self):
        path=PACKAGE/'evidence/visibility'
        declaration=json.loads((path/'declaration.json').read_text())
        for name,expected in declaration['source_sha256'].items():
            self.assertEqual(ref.digest(PACKAGE/name),expected)
        status=json.loads((path/'status.json').read_text())
        score=json.loads((path/'scoring.json').read_text())
        self.assertTrue(status['all_declared_tests_pass'])
        self.assertTrue(score['offline_gate_pass'])
        self.assertEqual(len(json.loads((path/'negative-controls.json').read_text())),2)


@unittest.skipUnless(os.environ.get('RGBD_RESEARCH_CAPTURE_ROOT'),'External captures are optional and not published')
class ArchivedVerificationTests(unittest.TestCase):
    def test_fixed_grid_transport_reproduces_preserved_failure(self):
        payload,scene,_=load_capture(os.environ['RGBD_RESEARCH_CAPTURE_ROOT'],'static')
        result,_,_=ref.reconstruct(payload.frames,scene,payload.robot_snapshots)
        models=[dict(component=c['component'],**fixed_grid.qualify(c,payload.frames)) for c in result['components']]
        expected=json.loads((PACKAGE/'evidence/fixed_grid/static-blind.json').read_text())['alternative']
        self.assertEqual(ref.plain(models),expected)

    def test_visibility_replay_and_all_controls_are_deterministic(self):
        frozen=json.loads((PACKAGE/'evidence/visibility/blind-results.json').read_text())
        negative=json.loads((PACKAGE/'evidence/visibility/negative-controls.json').read_text())
        for case in ('static','v43'):
            with self.subTest(case=case):
                payload,scene,_=load_capture(os.environ['RGBD_RESEARCH_CAPTURE_ROOT'],case)
                result,clusters,envelopes=ref.reconstruct(payload.frames,scene,payload.robot_snapshots)
                result.pop('timing_ms')
                self.assertEqual(ref.plain(result),frozen[case]['reference'])
                predictions=[dict(component=c['component'],**visibility.qualify(c,payload.frames)) for c in result['components']]
                self.assertEqual(ref.plain(predictions),frozen[case]['models'])
                candidate=next(c for c in result['components'] if c.get('hypothesis'))
                points=sorted(clusters,key=len,reverse=True)[candidate['component']]
                # The pre-existing XYZ-median bug must not reappear under transport.
                shifted=ref.hypothesis(points+[.8,-.5,0.])
                np.testing.assert_allclose(shifted['dimensions'],candidate['hypothesis']['dimensions'],atol=1e-10,rtol=0)
                self.assertAlmostEqual(shifted['center'][2],candidate['hypothesis']['center'][2],places=10)
                for name,(p,frames) in controls.make(payload.frames,points,candidate['hypothesis'],payload.robot_snapshots).items():
                    original=ref.qualify(p,frames,envelopes)
                    model=visibility.qualify(original,frames)
                    self.assertEqual(ref.plain(dict(reference=original,model=model)),negative[case][name])
                    self.assertNotEqual(model['status'],'qualified')
                    self.assertIsNone(model.get('geometry'))
                    if name=='missing_lower_boundary':
                        self.assertFalse(original['faces']['z-1']['qualified'])
                        self.assertIsNone(original['faces']['z-1']['outer_local_m'])


if __name__=='__main__':unittest.main()

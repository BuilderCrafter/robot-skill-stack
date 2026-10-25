"""Exact-cell, expiry, selection and resource-bound proofs for V3 research."""
from dataclasses import FrozenInstanceError
from itertools import product
import pickle
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import (
    GridSpec, OccupancySnapshot, OccupancyState, _readonly)
from robot_skill_stack.world.spatial_mapping.obstacles import (
    ObstacleExtractionSettings, ObstacleExtractor, _cuboids)
from scripts.obstacle_representation_v3 import (
    AXIS_ORDERS, PACKINGS, SELECTIONS, ExperimentalObstacleExtractor,
    MotionContext, audit_bounds, connected_components, pack_cells, prepare_evidence,
    public_motion_context)


def snapshot(mask, *, probabilities=None, observed_at=None, states=None, expired=None,
             timestamp=100., revision=7, generation=2, spec=None):
    mask=np.asarray(mask,bool)
    spec=spec or GridSpec((0.,0.,0.),tuple(np.asarray(mask.shape)*.01),.01)
    state=np.where(mask,OccupancyState.OCCUPIED,OccupancyState.UNKNOWN).astype(np.uint8) if states is None else np.asarray(states,np.uint8)
    probabilities=np.where(mask,.99,.5) if probabilities is None else np.asarray(probabilities,float)
    observed=state!=OccupancyState.UNKNOWN
    observed_at=np.where(observed,100.,np.nan) if observed_at is None else np.asarray(observed_at,float)
    expired=np.zeros(mask.shape,bool) if expired is None else np.asarray(expired,bool)
    return OccupancySnapshot(spec,*map(_readonly,(state,probabilities,observed,observed_at,expired)),
        timestamp,revision,generation,MappingProxyType(dict(status='ok')))


def materialize(packet,spec):
    indices=np.indices(spec.shape).reshape(3,-1).T
    centers=spec.indices_to_centers(indices)
    counts=np.zeros(len(indices),int)
    for box in packet.boxes:
        lower=np.asarray(box.center)-np.asarray(box.dimensions)/2
        upper=np.asarray(box.center)+np.asarray(box.dimensions)/2
        counts+=np.all((centers>=lower-1e-12)&(centers<upper-1e-12),axis=1)
    return counts.reshape(spec.shape)


def signature(packet):
    return [(b.id,b.center,b.dimensions,b.expires_at,b.voxel_count) for b in packet.boxes]


def heterogeneous_t():
    mask=np.zeros((12,2,12),bool)
    mask[0,0,0:3]=True;mask[1,0,1]=True
    mask[5:8,0,5]=True;mask[6,0,6]=True
    return mask


class ObstacleRepresentationV3Tests(unittest.TestCase):
    def test_public_context_matches_native_and_serialized_offline_origins(self):
        def part(position):
            transform = np.eye(4)
            transform[:3, 3] = position
            # Reading vertices would make offline/native definitions differ.
            return SimpleNamespace(world_from_local=transform, mesh=object())
        first, second = part([.1, .2, .3]), part([.4, .5, .6])
        snapshots = {'z_camera': SimpleNamespace(parts=(second, first)),
                     'a_camera': SimpleNamespace(parts=(first, first))}
        values = dict(goal=(.45, .25, .35), corridor=((.45, -.25, .35), (.45, .25, .35)),
                      relevance_radius_m=.20)
        native = public_motion_context(snapshots, **values)
        offline = public_motion_context(pickle.loads(pickle.dumps(snapshots)), **values)
        reversed_mapping = public_motion_context(dict(reversed(list(snapshots.items()))), **values)
        self.assertEqual(native, offline)
        self.assertEqual(native, reversed_mapping)
        self.assertEqual(native.robot_points, ((.1, .2, .3), (.4, .5, .6)))
        self.assertEqual(native.goal, values['goal'])
        with self.assertRaises(FrozenInstanceError):
            native.robot_points = ()

    def test_public_context_allows_explicit_goal_without_robot_and_rejects_bad_transform(self):
        self.assertEqual(public_motion_context(None, goal=(.1, .2, .3)).robot_points, ())
        with self.assertRaises(ValueError):
            public_motion_context({})
        with self.assertRaises(ValueError):
            public_motion_context({'camera': SimpleNamespace(parts=(
                SimpleNamespace(world_from_local=np.eye(3)),))}, goal=(0., 0., 0.))

    def assert_exact(self,mask,bounds,*,omitted=0):
        audit=audit_bounds(mask,bounds)
        self.assertEqual(audit['incorrect_covered_cells'],0)
        self.assertEqual(audit['overlapping_cells'],0)
        self.assertEqual(audit['out_of_bounds_boxes'],0)
        self.assertEqual(audit['omitted_cells'],omitted)
        self.assertEqual(audit['covered_voxels'],int(mask.sum())-omitted)

    def test_baseline_values_match_production_for_budgets_and_support_modes(self):
        random=np.random.default_rng(327)
        mask=random.random((12,11,8))<.23
        probabilities=np.where(random.random(mask.shape)<.5,.8,.99)
        source=snapshot(mask,probabilities=probabilities)
        for temporal,budget in product((False,True),(16,32,64,128)):
            with self.subTest(temporal=temporal,budget=budget):
                settings=ObstacleExtractionSettings(budget,0.)
                expected=ObstacleExtractor(settings,temporal_probabilities=temporal).extract(source,now=100.2,evidence_ttl=5.)
                actual=ExperimentalObstacleExtractor(settings,temporal_probabilities=temporal).extract(source,now=100.2,evidence_ttl=5.)
                self.assertEqual(signature(actual),signature(expected))
                for key in ('occupied_voxels','supported_voxels','spatial_supported_voxels','strong_sparse_retained',
                            'candidate_boxes','represented_voxels','omitted_voxels','selected_floor_boxes','selected_hazard_boxes'):
                    self.assertEqual(actual.diagnostics[key],expected.diagnostics[key],key)

    def test_all_six_sparse_axis_orders_match_transposed_production_packing(self):
        random=np.random.default_rng(891)
        for trial in range(4):
            mask=random.random((6,5,7))<.4
            for order in AXIS_ORDERS:
                with self.subTest(trial=trial,order=order):
                    permutation=tuple('xyz'.index(axis) for axis in reversed(order))
                    reference,_=_cuboids(mask.transpose(permutation))
                    restored=np.empty_like(reference)
                    for temporary,original in enumerate(permutation):
                        restored[:,2*original:2*original+2]=reference[:,2*temporary:2*temporary+2]
                    actual=pack_cells(mask,'axis_'+order)
                    self.assertEqual(sorted(map(tuple,actual.bounds)),sorted(map(tuple,restored)))
                    self.assert_exact(mask,actual.bounds)

    def test_greedy_and_component_are_genuine_improvements_over_global_axis_orders(self):
        mask=heterogeneous_t()
        self.assertEqual(len(pack_cells(mask,'baseline').bounds),5)
        self.assertEqual(len(pack_cells(mask,'axis_best').bounds),5)
        for policy in ('greedy','component'):
            result=pack_cells(mask,policy)
            self.assertEqual(len(result.bounds),4)
            self.assert_exact(mask,result.bounds)
        components=connected_components(np.argwhere(mask),6)
        self.assertEqual(len(components),2)
        # Each component is nonrectangular, so each requires at least two boxes.
        self.assertTrue(all(len(c)<np.prod(c.max(0)-c.min(0)+1) for c in components))

    def test_every_algorithm_preserves_unknown_hole_and_known_free_gap(self):
        mask=np.zeros((11,9,5),bool)
        mask[1:4,1:8,1:4]=True;mask[5:9,1:3,1:4]=True;mask[7:9,1:8,1:4]=True
        states=np.where(mask,OccupancyState.OCCUPIED,OccupancyState.UNKNOWN).astype(np.uint8)
        states[4,:,:]=OccupancyState.FREE
        source=snapshot(mask,states=states)
        for policy in PACKINGS:
            with self.subTest(policy=policy):
                extractor=ExperimentalObstacleExtractor(ObstacleExtractionSettings(128),packing=policy)
                packet=extractor.extract(source,now=100.,evidence_ttl=5.)
                self.assert_exact(mask,extractor.last_selected_bounds)
                np.testing.assert_array_equal(materialize(packet,source.spec),mask.astype(int))

    def test_supported_cells_keep_original_radius_two_and_probability_semantics(self):
        mask=np.zeros((12,3,3),bool);mask[[0,2,4],1,1]=True;mask[10,1,1]=True
        probabilities=np.where(mask,.80,.5);probabilities[10,1,1]=.95
        source=snapshot(mask,probabilities=probabilities)
        spatial=prepare_evidence(source,now=100.,evidence_ttl=5.,temporal_probabilities=False)
        temporal=prepare_evidence(source,now=100.,evidence_ttl=5.,temporal_probabilities=True)
        self.assertEqual(int(spatial.supported.sum()),3)
        self.assertEqual(int(temporal.supported.sum()),4)
        self.assertEqual(temporal.counts['strong_sparse_retained'],1)
        self.assertFalse(temporal.supported[1,1,1])

    def test_free_unknown_expired_future_and_nan_cells_never_become_boxes(self):
        mask=np.ones((8,1,1),bool)
        states=np.full(mask.shape,OccupancyState.OCCUPIED,np.uint8)
        states[1]=OccupancyState.FREE;states[2]=OccupancyState.UNKNOWN
        times=np.full(mask.shape,100.);times[4]=101.;times[5]=np.nan;times[6]=90.
        expired=np.zeros(mask.shape,bool);expired[3]=True
        source=snapshot(mask,states=states,observed_at=times,expired=expired)
        for policy in PACKINGS:
            packet=ExperimentalObstacleExtractor(packing=policy).extract(source,now=100.,evidence_ttl=5.)
            actual=materialize(packet,source.spec)
            expected=np.zeros(mask.shape,int);expected[0]=expected[7]=1
            np.testing.assert_array_equal(actual,expected)

    def test_oldest_expiry_and_source_identity_survive_then_expire_exactly(self):
        mask=np.ones((3,1,1),bool);times=np.array([100.,101.,102.]).reshape(mask.shape)
        source=snapshot(mask,observed_at=times,timestamp=102.,revision=21,generation=4)
        before_states=source.states.copy()
        for policy in PACKINGS:
            extractor=ExperimentalObstacleExtractor(packing=policy)
            first=extractor.extract(source,now=102.,evidence_ttl=5.)
            self.assertEqual((first.timestamp,first.revision,first.generation),(102.,21,4))
            self.assertTrue(all(box.expires_at==105. for box in first.boxes))
            later=extractor.extract(source,now=105.,evidence_ttl=5.)
            actual=materialize(later,source.spec)
            np.testing.assert_array_equal(actual.ravel(),[0,1,1])
            self.assertTrue(all(box.expires_at==106. for box in later.boxes))
        np.testing.assert_array_equal(source.states,before_states)

    def test_partial_boundary_cells_are_clipped_without_padding(self):
        spec=GridSpec((0.,0.,0.),(.025,.015,.013),.01)
        source=snapshot(np.ones(spec.shape,bool),spec=spec)
        for policy in PACKINGS:
            packet=ExperimentalObstacleExtractor(packing=policy).extract(source,now=100.,evidence_ttl=5.)
            self.assertEqual(len(packet.boxes),1)
            np.testing.assert_allclose(packet.boxes[0].dimensions,spec.bounds_max,atol=1e-15)
            self.assertEqual(packet.boxes[0].voxel_count,12)
            self.assertEqual(packet.diagnostics['padding_m'],0.)

    def test_greedy_inspection_budget_falls_back_without_losing_geometry(self):
        mask=np.ones((4,4,4),bool)
        result=pack_cells(mask,'greedy',greedy_inspection_budget=1)
        self.assertTrue(result.diagnostics['greedy_fallback'])
        self.assertLessEqual(result.diagnostics['greedy_inspected_cells'],1)
        self.assert_exact(mask,result.bounds)

    def test_component_and_greedy_cell_caps_are_explicit_exact_fallbacks(self):
        mask=heterogeneous_t()
        for policy in ('component','greedy'):
            result=pack_cells(mask,policy,max_component_cells=2)
            self.assertIsNotNone(result.diagnostics['fallback'])
            self.assert_exact(mask,result.bounds)

    def test_run_cap_reports_the_exact_number_of_omitted_cells(self):
        mask=np.zeros((4,4,9),bool);mask[:,:,::2]=True
        result=pack_cells(mask,'axis_zyx',max_runs=2)
        self.assertGreater(result.dropped_voxels,0)
        self.assert_exact(mask,result.bounds,omitted=result.dropped_voxels)

    def test_face_components_do_not_treat_diagonal_contact_as_a_box_bridge(self):
        cells=np.array([[0,0,0],[1,1,1],[2,2,2]])
        self.assertEqual(len(connected_components(cells,6)),3)
        self.assertEqual(len(connected_components(cells,26)),1)
        with self.assertRaises(ValueError):connected_components(cells,6,max_cells=2)

    def test_coverage_has_a_genuine_floor_priority_tradeoff(self):
        mask=np.zeros((9,5,6),bool);mask[:6,:3,0]=True;mask[7,4,2:5]=True
        source=snapshot(mask)
        baseline=ExperimentalObstacleExtractor(ObstacleExtractionSettings(1,0.)).extract(source,now=100.,evidence_ttl=5.)
        coverage=ExperimentalObstacleExtractor(ObstacleExtractionSettings(1,0.),selection='coverage').extract(source,now=100.,evidence_ttl=5.)
        self.assertEqual(baseline.diagnostics['selected_hazard_boxes'],1)
        self.assertEqual(coverage.diagnostics['selected_floor_boxes'],1)
        self.assertGreater(coverage.diagnostics['represented_voxels'],baseline.diagnostics['represented_voxels'])

    def test_diversity_retains_distant_observed_hazard_instead_of_nearby_tie(self):
        mask=np.zeros((15,10,3),bool);mask[[0,2,4],0,1]=True;mask[14,9,1]=True
        source=snapshot(mask)
        baseline=ExperimentalObstacleExtractor(ObstacleExtractionSettings(2)).extract(source,now=100.,evidence_ttl=5.)
        diverse=ExperimentalObstacleExtractor(ObstacleExtractionSettings(2),selection='diversity').extract(source,now=100.,evidence_ttl=5.)
        self.assertGreater(max(box.center[0] for box in diverse.boxes),max(box.center[0] for box in baseline.boxes))
        self.assertEqual(sum(box.voxel_count for box in diverse.boxes),2)

    def test_motion_selection_reserves_outside_evidence_and_reports_omissions(self):
        mask=np.zeros((24,12,3),bool);mask[[0,2,4,6,8],0,1]=True;mask[[10,16,22],8,1]=True
        source=snapshot(mask)
        context=MotionContext(robot_points=((.005,.005,.015),),goal=(.085,.005,.015),
                              corridor=((.005,.005,.015),(.085,.005,.015)),relevance_radius_m=.025)
        packet=ExperimentalObstacleExtractor(ObstacleExtractionSettings(4),selection='motion',motion_context=context).extract(source,now=100.,evidence_ttl=5.)
        self.assertEqual(packet.diagnostics['outside_region_candidate_boxes'],3)
        self.assertEqual(packet.diagnostics['selected_outside_region_boxes'],1)
        self.assertEqual(packet.diagnostics['omitted_outside_region_voxels'],2)
        self.assertEqual(len(packet.boxes),4)
        with self.assertRaises(ValueError):ExperimentalObstacleExtractor(selection='motion')

    def test_ids_are_deterministic_and_public_records_are_immutable(self):
        source=snapshot(heterogeneous_t())
        for policy,selection in product(PACKINGS,SELECTIONS):
            context=MotionContext(goal=(.03,.01,.03)) if selection=='motion' else None
            first=ExperimentalObstacleExtractor(packing=policy,selection=selection,motion_context=context)
            second=ExperimentalObstacleExtractor(packing=policy,selection=selection,motion_context=context)
            a=first.extract(source,now=100.,evidence_ttl=5.)
            b=second.extract(source,now=100.,evidence_ttl=5.)
            self.assertEqual(signature(a),signature(b))
            with self.assertRaises(ValueError):first.last_candidate_bounds.setflags(write=True)
            with self.assertRaises(TypeError):a.diagnostics['padding_m']=1.
        context=MotionContext(goal=[.1,.2,.3])
        with self.assertRaises(FrozenInstanceError):context.goal=(0.,0.,0.)

    def test_empty_and_invalid_experiments_do_not_invent_evidence(self):
        source=snapshot(np.zeros((3,3,3),bool),timestamp=None)
        for policy in PACKINGS:
            packet=ExperimentalObstacleExtractor(packing=policy).extract(source,now=100.,evidence_ttl=5.)
            self.assertEqual(packet.boxes,());self.assertIsNone(packet.timestamp)
        with self.assertRaises(ValueError):pack_cells(np.zeros((2,2),bool),'greedy')
        with self.assertRaises(ValueError):ExperimentalObstacleExtractor(packing='unknown')
        with self.assertRaises(ValueError):MotionContext()
        with self.assertRaises(ValueError):MotionContext(goal=(float('nan'),0.,0.))
        with self.assertRaises(ValueError):prepare_evidence(source,now=100.,evidence_ttl=0.)


if __name__=='__main__':
    unittest.main()

"""SDK-independent, exact occupied-cell representation research.

The production eligibility and support rule is retained verbatim. Experiments
change only exact partitions of that support and bounded subset selection. No
completion, dilation, padding, scene labels or future collision outcomes enter
this module. All existing packet records and original cell expiries are reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math
import time
from types import MappingProxyType

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyState
from robot_skill_stack.world.spatial_mapping.obstacles import (
    ObstacleBox, ObstacleExtractionSettings, ObstaclePacket, _cuboids, _supported)


AXIS_ORDERS = tuple(''.join(order) for order in permutations('xyz'))
PACKINGS = ('baseline', *(f'axis_{order}' for order in AXIS_ORDERS), 'axis_best', 'greedy', 'component')
SELECTIONS = ('baseline', 'coverage', 'diversity', 'motion')
MAX_COMPONENT_CELLS = 20_000
MAX_RUNS = 100_000
GREEDY_INSPECTION_BUDGET = 2_000_000
GREEDY_FACE_CHECK_BUDGET = 20_000


def _readonly(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _points(value, name):
    result = tuple(tuple(map(float, point)) for point in value)
    if any(len(point) != 3 or not all(map(math.isfinite, point)) for point in result):
        raise ValueError(name+' requires finite XYZ tuples')
    return result


@dataclass(frozen=True)
class MotionContext:
    """Explicit research-only public state; never labels or oracle geometry.

    ``corridor`` is an intended TCP waypoint polyline. ``robot_points`` may be
    observed link centers, and ``goal`` is the requested target position. The
    radius classifies candidate centers for ranking/reserve diagnostics only;
    it never changes an obstacle's geometry. A fixed START/GOAL context must be
    reported as fixed, not as continuously updated robot state.
    """
    robot_points: tuple = ()
    goal: tuple | None = None
    corridor: tuple = ()
    relevance_radius_m: float = .20

    def __post_init__(self):
        object.__setattr__(self, 'robot_points', _points(self.robot_points, 'robot_points'))
        object.__setattr__(self, 'corridor', _points(self.corridor, 'corridor'))
        if self.goal is not None:
            object.__setattr__(self, 'goal', _points((self.goal,), 'goal')[0])
        radius = float(self.relevance_radius_m)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('relevance_radius_m must be finite and positive')
        if not self.robot_points and not self.corridor and self.goal is None:
            raise ValueError('Motion context requires public robot, corridor or goal points')
        object.__setattr__(self, 'relevance_radius_m', radius)


def public_motion_context(robot_snapshots, *, goal=None, corridor=(), relevance_radius_m=.20):
    """Build the same numeric motion context offline and in the mapping worker.

    Inputs are exposure-matched public robot mesh snapshots, ordered by camera
    ID. A part contributes its local-frame origin transformed into the world,
    not its bounding-box center. Identical origins are retained once, in first
    occurrence order. This helper never reads mesh vertices or fixture labels;
    exposure matching remains the caller's existing acquisition/filter check.
    """
    points, seen = [], set()
    for camera in sorted(robot_snapshots or {}):
        for part in robot_snapshots[camera].parts:
            transform = np.asarray(part.world_from_local, dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('Public robot origins require finite4x4 transforms')
            point = tuple(map(float, transform[:3, 3]))
            if point not in seen:
                seen.add(point)
                points.append(point)
    return MotionContext(robot_points=tuple(points), goal=goal, corridor=corridor,
                         relevance_radius_m=relevance_radius_m)


@dataclass(frozen=True)
class Evidence:
    eligible: np.ndarray
    spatial: np.ndarray
    supported: np.ndarray
    counts: object

    @property
    def cells(self):
        return _readonly(np.argwhere(self.supported), np.int64)


@dataclass(frozen=True)
class PackingResult:
    bounds: np.ndarray
    dropped_voxels: int
    diagnostics: object


def prepare_evidence(snapshot, *, now, evidence_ttl, temporal_probabilities=True):
    """Exactly the baseline OCCUPIED/observed/expiry/support eligibility rule."""
    now, evidence_ttl = float(now), float(evidence_ttl)
    if (not math.isfinite(now) or now < 0 or not math.isfinite(evidence_ttl) or evidence_ttl <= 0
            or type(temporal_probabilities) is not bool):
        raise ValueError('Finite extraction time, positive TTL and boolean temporal support required')
    observed_at = snapshot.last_observed
    eligible = ((snapshot.states == OccupancyState.OCCUPIED) & snapshot.observed &
        ~snapshot.expired & np.isfinite(observed_at) & (observed_at <= now) &
        (now-observed_at < evidence_ttl))
    spatial = _supported(eligible)
    supported = spatial.copy()
    strong_count = 0
    if temporal_probabilities:
        strong = eligible & ~spatial & (snapshot.probabilities >= .95)
        strong_count = int(strong.sum())
        supported |= strong
    return Evidence(_readonly(eligible), _readonly(spatial), _readonly(supported), _freeze(dict(
        occupied_voxels=int(eligible.sum()), supported_voxels=int(supported.sum()),
        spatial_supported_voxels=int(spatial.sum()), strong_sparse_retained=strong_count,
        rejected_sparse_voxels=int(eligible.sum()-supported.sum()),
        temporal_probabilities=temporal_probabilities,
        temporal_support_probability=.95 if temporal_probabilities else None)))


def connected_components(cells, adjacency=6, *, max_cells=MAX_COMPONENT_CELLS):
    """Deterministic sparse components; 6/26 adjacency never adds cell geometry."""
    if type(max_cells) is not int or max_cells < 1:
        raise ValueError('Component cell limit must be a positive integer')
    values = np.asarray(cells)
    if values.ndim == 3 and values.dtype == np.bool_:
        values = np.argwhere(values)
    if values.size == 0:
        return ()
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in 'iu':
        raise ValueError('Components require integer XYZ cells or a boolean grid')
    if adjacency not in (6, 26):
        raise ValueError('Component adjacency must be6 or26')
    if len(values) > max_cells:
        raise ValueError('Component cell limit exceeded')
    offsets = tuple(offset for offset in product((-1, 0, 1), repeat=3)
                    if sum(abs(item) for item in offset) in ((1,) if adjacency == 6 else (1, 2, 3)))
    ordered = sorted(set(map(tuple, values.tolist())))
    remaining = set(ordered)
    result = []
    for seed in ordered:
        if seed not in remaining:
            continue
        remaining.remove(seed)
        pending, found = [seed], []
        while pending:
            cell = pending.pop()
            found.append(cell)
            for offset in offsets:
                neighbor = tuple(a+b for a, b in zip(cell, offset))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        result.append(_readonly(sorted(found), np.int64))
    return tuple(result)


def _merge_runs(starts, stops):
    """The baseline exact interval-identity merges, on any permuted axes."""
    if not len(starts):
        return np.empty((0, 6), np.int64)
    order = np.lexsort((starts[:, 1], stops[:, 2], starts[:, 2], starts[:, 0]))
    starts, stops = starts[order], stops[order]
    same = ((starts[1:, 0] == starts[:-1, 0]) & (starts[1:, 2] == starts[:-1, 2]) &
            (stops[1:, 2] == stops[:-1, 2]) & (starts[1:, 1] == starts[:-1, 1]+1))
    first = np.r_[0, np.flatnonzero(~same)+1]
    last = np.r_[first[1:]-1, len(starts)-1]
    boxes = np.column_stack((starts[first, 0], starts[first, 0]+1,
        starts[first, 1], starts[last, 1]+1, starts[first, 2], stops[first, 2]))
    order = np.lexsort((boxes[:, 0], boxes[:, 5], boxes[:, 4], boxes[:, 3], boxes[:, 2]))
    boxes = boxes[order]
    same = np.all(boxes[1:, 2:] == boxes[:-1, 2:], axis=1) & (boxes[1:, 0] == boxes[:-1, 1])
    first = np.r_[0, np.flatnonzero(~same)+1]
    last = np.r_[first[1:]-1, len(boxes)-1]
    result = boxes[first].copy()
    result[:, 1] = boxes[last, 1]
    return result


def _axis_cells(cells, order, max_runs):
    """Sparse exact run/merge packing; order names mean run→merge→merge."""
    if not len(cells):
        return np.empty((0, 6), np.int64), 0
    permutation = tuple('xyz'.index(axis) for axis in reversed(order))
    points = np.asarray(cells, np.int64)[:, permutation]
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    same = np.all(points[1:, :2] == points[:-1, :2], axis=1) & (points[1:, 2] == points[:-1, 2]+1)
    first = np.r_[0, np.flatnonzero(~same)+1]
    last = np.r_[first[1:]-1, len(points)-1]
    starts, stops = points[first].copy(), points[last].copy()
    stops[:, 2] += 1
    lengths = stops[:, 2]-starts[:, 2]
    dropped = 0
    if len(starts) > max_runs:
        chosen = np.lexsort((starts[:, 2], starts[:, 1], starts[:, 0], -lengths))[:max_runs]
        dropped = int(lengths.sum()-lengths[chosen].sum())
        starts, stops = starts[chosen], stops[chosen]
    packed = _merge_runs(starts, stops)
    result = np.empty_like(packed)
    for temporary, original in enumerate(permutation):
        result[:, 2*original:2*original+2] = packed[:, 2*temporary:2*temporary+2]
    return result, dropped


def _packing_key(bounds, dropped, order):
    # Omitted cells take precedence over compactness under a pathological run
    # cap; an algorithm may not improve its count by quietly dropping support.
    return dropped, len(bounds), order


def _best_axis(cells, max_runs):
    choices = []
    for order in AXIS_ORDERS:
        bounds, dropped = _axis_cells(cells, order, max_runs)
        choices.append((bounds, dropped, order))
    return min(choices, key=lambda value: _packing_key(*value)), {value[2]: len(value[0]) for value in choices}


def _greedy(mask, cells, max_runs, inspection_budget):
    working = np.array(mask, dtype=bool, copy=True)
    result, inspected, face_checks, exhausted, isolated = [], 0, 0, False, 0
    unit_axes = np.eye(3, dtype=np.int64)
    for seed in cells:
        if not working[tuple(seed)]:
            continue
        best = np.column_stack((seed, seed+1)).ravel()
        neighbors = [seed+unit_axes[axis] for axis in range(3)]
        if not any(neighbor[axis] < mask.shape[axis] and working[tuple(neighbor)]
                   for axis, neighbor in enumerate(neighbors)):
            # The seed is lexically first remaining: negative neighbors are
            # already absent. No exact non-unit box can contain this seed.
            result.append(best)
            working[tuple(seed)] = False
            isolated += 1
            continue
        best_key = (-1, tuple(best))
        for axes in permutations(range(3)):
            bounds = np.column_stack((seed, seed+1)).ravel()
            for axis in axes:
                while bounds[2*axis+1] < mask.shape[axis]:
                    slices = [slice(int(bounds[2*a]), int(bounds[2*a+1])) for a in range(3)]
                    slices[axis] = slice(int(bounds[2*axis+1]), int(bounds[2*axis+1]+1))
                    cost = math.prod(s.stop-s.start for s in slices)
                    if inspected+cost > inspection_budget or face_checks >= GREEDY_FACE_CHECK_BUDGET:
                        exhausted = True
                        break
                    inspected += cost
                    face_checks += 1
                    if not working[tuple(slices)].all():
                        break
                    bounds[2*axis+1] += 1
                if exhausted:
                    break
            volume = math.prod(int(bounds[2*a+1]-bounds[2*a]) for a in range(3))
            key = (-volume, tuple(bounds))
            if key < best_key:
                best, best_key = bounds.copy(), key
            if exhausted:
                break
        result.append(best)
        working[tuple(slice(int(best[2*a]), int(best[2*a+1])) for a in range(3))] = False
        if exhausted:
            break
    dropped, fallback_count = 0, 0
    if exhausted and working.any():
        remaining = np.argwhere(working)
        fallback_count = len(remaining)
        following, dropped = _axis_cells(remaining, 'zyx', max_runs)
        result.extend(following)
    return np.asarray(result, dtype=np.int64).reshape(-1, 6), dropped, dict(
        greedy_inspected_cells=inspected, greedy_face_checks=face_checks,
        greedy_inspection_budget=inspection_budget, greedy_fallback=exhausted,
        greedy_face_check_budget=GREEDY_FACE_CHECK_BUDGET,
        greedy_isolated_remaining_seeds=isolated, fallback_remaining_voxels=fallback_count)


def pack_cells(mask, packing='baseline', *, max_runs=MAX_RUNS,
               greedy_inspection_budget=GREEDY_INSPECTION_BUDGET,
               max_component_cells=MAX_COMPONENT_CELLS):
    """Bounded exact partitions; all omissions/fallbacks are explicit."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != np.bool_ or packing not in PACKINGS:
        raise ValueError('A boolean XYZ grid and a known packing policy are required')
    if any(type(value) is not int or value < 1 for value in (max_runs, greedy_inspection_budget, max_component_cells)):
        raise ValueError('Packing resource bounds must be positive integers')
    cells = np.argwhere(mask)
    details = dict(packing=packing, axis_order_convention='run, then first merge, then second merge',
                   fallback=None, max_runs=max_runs, sparse_algorithm_cell_limit=max_component_cells)
    if packing == 'baseline':
        bounds, dropped = _cuboids(mask, max_runs=max_runs)
        details['chosen_axis'] = 'zyx'
    elif packing.startswith('axis_') and packing != 'axis_best':
        order = packing.removeprefix('axis_')
        bounds, dropped = _axis_cells(cells, order, max_runs)
        details['chosen_axis'] = order
    elif packing == 'axis_best':
        (bounds, dropped, order), counts = _best_axis(cells, max_runs)
        details.update(chosen_axis=order, axis_candidate_counts=counts)
    elif packing == 'greedy' and len(cells) > max_component_cells:
        bounds, dropped = _axis_cells(cells, 'zyx', max_runs)
        details.update(fallback='greedy_seed_cell_limit_to_sparse_zyx')
    elif packing == 'greedy':
        bounds, dropped, extra = _greedy(mask, cells, max_runs, greedy_inspection_budget)
        details.update(extra)
    elif len(cells) > max_component_cells:
        bounds, dropped = _axis_cells(cells, 'zyx', max_runs)
        details.update(fallback='component_cell_limit_to_sparse_zyx', component_cell_limit=max_component_cells)
    else:
        components = connected_components(cells, max_cells=max_component_cells)
        pieces, dropped, choices = [], 0, {order: 0 for order in AXIS_ORDERS}
        for component in components:
            if len(component) == 1:
                bounds = np.column_stack((component[0], component[0]+1)).ravel()[None, :]
                loss, order = 0, 'xyz'
            else:
                (bounds, loss, order), _ = _best_axis(component, max_runs)
            pieces.append(bounds)
            dropped += loss
            choices[order] += 1
        bounds = np.concatenate(pieces) if pieces else np.empty((0, 6), np.int64)
        details.update(component_adjacency='face6', component_count=len(components),
                       component_chosen_axis_counts=choices)
    return PackingResult(_readonly(bounds, np.int64), int(dropped), _freeze(details))


def audit_bounds(mask, bounds):
    """Independent cell-level coverage check, intended for offline evaluation."""
    mask, bounds = np.asarray(mask, bool), np.asarray(bounds, np.int64).reshape(-1, 6)
    covered = np.zeros(mask.shape, bool)
    incorrect, overlapping, outside = 0, 0, 0
    for box in bounds:
        lower, upper = box[[0, 2, 4]], box[[1, 3, 5]]
        if np.any(lower < 0) or np.any(upper > mask.shape) or np.any(upper <= lower):
            outside += 1
            continue
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        incorrect += int((~mask[slices]).sum())
        overlapping += int(covered[slices].sum())
        covered[slices] = True
    return dict(covered_voxels=int(covered.sum()), incorrect_covered_cells=incorrect,
        overlapping_cells=overlapping, out_of_bounds_boxes=outside,
        omitted_cells=int(np.count_nonzero(mask & ~covered)))


def _lexical_order(bounds, volumes, floor, *, floor_first_key=True):
    if floor_first_key:
        return np.lexsort((bounds[:, 4], bounds[:, 2], bounds[:, 0], -volumes, floor))
    return np.lexsort((bounds[:, 4], bounds[:, 2], bounds[:, 0], -volumes))


def _diverse(bounds, volumes, floor, budget, *, allowed=None, centers=None):
    if budget <= 0 or not len(bounds):
        return np.empty(0, np.int64)
    baseline = _lexical_order(bounds, volumes, floor)
    ranking = np.empty(len(bounds), np.int64)
    ranking[baseline] = np.arange(len(bounds))
    if centers is None:
        centers = (bounds[:, [0, 2, 4]]+bounds[:, [1, 3, 5]])/2
    available = np.ones(len(bounds), bool) if allowed is None else np.array(allowed, bool, copy=True)
    distances = np.full(len(bounds), np.inf)
    selected = []
    for phase in (False, True):
        candidates = available & (floor == phase)
        while candidates.any() and len(selected) < budget:
            ids = np.flatnonzero(candidates)
            chosen = ids[np.lexsort((ranking[ids], -distances[ids]))[0]]
            selected.append(int(chosen))
            available[chosen] = False
            candidates[chosen] = False
            distances = np.minimum(distances, np.sum((centers-centers[chosen])**2, axis=1))
        if len(selected) >= budget:
            break
    return np.asarray(selected, np.int64)


def _motion_distances(centers, context):
    points = list(context.robot_points)+list(context.corridor)
    if context.goal is not None:
        points.append(context.goal)
    distances = np.min(np.linalg.norm(centers[:, None, :]-np.asarray(points)[None], axis=2), axis=1)
    for first, last in zip(context.corridor, context.corridor[1:]):
        first, last = np.asarray(first), np.asarray(last)
        direction = last-first
        denominator = float(direction@direction)
        if denominator <= 1e-20:
            continue
        fraction = np.clip((centers-first)@direction/denominator, 0., 1.)
        distances = np.minimum(distances, np.linalg.norm(centers-(first+fraction[:, None]*direction), axis=1))
    return distances


def select_bounds(snapshot, bounds, *, settings, selection='baseline', motion_context=None):
    """Select exact existing boxes; policies never create or resize geometry."""
    bounds = np.asarray(bounds, np.int64).reshape(-1, 6)
    if selection not in SELECTIONS:
        raise ValueError('Unknown selection policy')
    if selection == 'motion' and not isinstance(motion_context, MotionContext):
        raise ValueError('Motion selection requires an explicit immutable MotionContext')
    sizes = bounds[:, [1, 3, 5]]-bounds[:, [0, 2, 4]]
    volumes = np.prod(sizes, axis=1, dtype=np.int64)
    minimum = np.asarray(snapshot.spec.bounds_min)+bounds[:, [0, 2, 4]]*snapshot.spec.voxel_size
    maximum = np.minimum(np.asarray(snapshot.spec.bounds_min)+bounds[:, [1, 3, 5]]*snapshot.spec.voxel_size,
                         snapshot.spec.bounds_max)
    centers = (minimum+maximum)/2
    floor = np.zeros(len(bounds), bool)
    if settings.support_plane_z is not None:
        tops = np.minimum(snapshot.spec.bounds_min[2]+bounds[:, 5]*snapshot.spec.voxel_size,
                          snapshot.spec.bounds_max[2])
        floor = tops <= settings.support_plane_z+snapshot.spec.voxel_size+1e-12
    details = dict(selection=selection, candidate_boxes=len(bounds),
        floor_candidate_boxes=int(floor.sum()), hazard_candidate_boxes=int((~floor).sum()),
        floor_candidate_voxels=int(volumes[floor].sum()),
        selection_distance_scope='Candidate centers only; no geometry inflation or oracle labels')
    budget = settings.max_obstacles
    if selection == 'baseline':
        selected = _lexical_order(bounds, volumes, floor)[:budget]
    elif selection == 'coverage':
        selected = _lexical_order(bounds, volumes, floor, floor_first_key=False)[:budget]
        details['coverage_objective'] = 'Maximum retained cells for this disjoint partition; floor has equal weight, so hazard coverage may decline.'
    elif selection == 'diversity':
        selected = _diverse(bounds, volumes, floor, budget, centers=centers)
        details['diversity_objective'] = 'Hazards before floor, then farthest candidate center from selected centers; volume/lexical ties.'
    else:
        distances = _motion_distances(centers, motion_context) if len(bounds) else np.empty(0)
        outside = distances > motion_context.relevance_radius_m
        reserve = max(1, math.ceil(budget/4))
        # Prioritize hazards consistently. In each class reserve outside-region
        # slots when evidence exists, then fill by public center relevance.
        chosen = []
        for phase in (False, True):
            remaining_slots = budget-len(chosen)
            if remaining_slots <= 0:
                break
            allowed = (floor == phase) & outside
            reserved = _diverse(bounds, volumes, floor, min(reserve, remaining_slots), allowed=allowed, centers=centers)
            chosen.extend(map(int, reserved))
            available = (floor == phase)
            available[chosen] = False
            ids = np.flatnonzero(available)
            order = np.lexsort((bounds[ids, 4], bounds[ids, 2], bounds[ids, 0], -volumes[ids], distances[ids]))
            chosen.extend(map(int, ids[order[:budget-len(chosen)]]))
        selected = np.asarray(chosen, np.int64)
        details.update(motion_context=dict(robot_points=motion_context.robot_points, goal=motion_context.goal,
            corridor=motion_context.corridor, relevance_radius_m=motion_context.relevance_radius_m),
            outside_region_reserve_slots=reserve, outside_region_candidate_boxes=int(outside.sum()),
            selected_outside_region_boxes=int(outside[selected].sum()),
            omitted_outside_region_voxels=int(volumes[outside].sum()-volumes[selected][outside[selected]].sum()))
    details.update(selected_floor_boxes=int(floor[selected].sum()), selected_hazard_boxes=int((~floor[selected]).sum()),
        represented_floor_voxels=int(volumes[selected][floor[selected]].sum()),
        omitted_floor_voxels=int(volumes[floor].sum()-volumes[selected][floor[selected]].sum()))
    return _readonly(selected, np.int64), _freeze(details)


class ExperimentalObstacleExtractor:
    """Worker-only prototype yielding the existing immutable obstacle packet."""
    def __init__(self, settings=None, *, temporal_probabilities=True, packing='baseline', selection='baseline',
                 motion_context=None, max_runs=MAX_RUNS, greedy_inspection_budget=GREEDY_INSPECTION_BUDGET,
                 max_component_cells=MAX_COMPONENT_CELLS):
        if packing not in PACKINGS or selection not in SELECTIONS:
            raise ValueError('Unknown experimental packing or selection')
        if type(temporal_probabilities) is not bool:
            raise ValueError('Temporal probability support must be boolean')
        if selection == 'motion' and not isinstance(motion_context, MotionContext):
            raise ValueError('Motion selection needs explicit public-state context')
        self.settings = settings or ObstacleExtractionSettings()
        self._temporal_probabilities = temporal_probabilities
        self.packing, self.selection, self.motion_context = packing, selection, motion_context
        self.max_runs, self.greedy_inspection_budget = max_runs, greedy_inspection_budget
        self.max_component_cells = max_component_cells
        self.last_candidate_bounds = _readonly(np.empty((0, 6), np.int64))
        self.last_selected_bounds = _readonly(np.empty((0, 6), np.int64))

    def extract(self, snapshot, *, now, evidence_ttl):
        started = time.perf_counter()
        evidence = prepare_evidence(snapshot, now=now, evidence_ttl=evidence_ttl,
                                    temporal_probabilities=self._temporal_probabilities)
        packed = pack_cells(evidence.supported, self.packing, max_runs=self.max_runs,
            greedy_inspection_budget=self.greedy_inspection_budget, max_component_cells=self.max_component_cells)
        indices, selection = select_bounds(snapshot, packed.bounds, settings=self.settings,
            selection=self.selection, motion_context=self.motion_context)
        self.last_candidate_bounds = packed.bounds
        self.last_selected_bounds = _readonly(packed.bounds[indices], np.int64)
        lower, upper = np.asarray(snapshot.spec.bounds_min), np.asarray(snapshot.spec.bounds_max)
        boxes = []
        for bounds in self.last_selected_bounds:
            x0, x1, y0, y1, z0, z1 = map(int, bounds)
            slices = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
            if not evidence.supported[slices].all():
                raise RuntimeError('Experimental packing attempted to cover ineligible geometry')
            minimum = lower+np.array([x0, y0, z0])*snapshot.spec.voxel_size
            maximum = np.minimum(lower+np.array([x1, y1, z1])*snapshot.spec.voxel_size, upper)
            volume = (x1-x0)*(y1-y0)*(z1-z0)
            oldest = float(snapshot.last_observed[slices].min())
            boxes.append(ObstacleBox(f'{x0}:{x1}:{y0}:{y1}:{z0}:{z1}',tuple((minimum+maximum)/2),
                tuple(maximum-minimum), oldest+float(evidence_ttl), int(volume)))
        diagnostics = dict(evidence.counts)
        diagnostics.update(packed.diagnostics)
        diagnostics.update(selection)
        represented = sum(box.voxel_count for box in boxes)
        diagnostics.update(proxy_count=len(boxes), represented_voxels=represented,
            omitted_voxels=evidence.counts['supported_voxels']-represented,
            run_budget_omitted_voxels=packed.dropped_voxels,
            truncated=len(packed.bounds)>self.settings.max_obstacles or packed.dropped_voxels>0,
            max_obstacles=self.settings.max_obstacles, minimum_component_voxels=3,
            support_radius_voxels=2, support_neighborhood='Chebyshev',
            support_plane_z=self.settings.support_plane_z, padding_m=0.,
            geometry_completion=False, selected_cell_bounds=tuple(map(tuple, self.last_selected_bounds.tolist())),
            extraction_recomputed=True, extract_seconds=time.perf_counter()-started)
        return ObstaclePacket(snapshot.generation, snapshot.revision, snapshot.timestamp,
                              float(now), tuple(boxes), _freeze(diagnostics))

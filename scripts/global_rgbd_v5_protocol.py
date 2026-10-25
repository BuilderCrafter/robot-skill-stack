"""Predeclared V5 source comparison, without Isaac imports or scene edits.

The native harness owns temporary physical fixtures and launch/measurement.
This helper only returns numeric fixture declarations and writes a run-local
profile copied from the authored two-camera baseline. It never sets a camera
viewpoint or changes shipped scene/profile files.
"""
from pathlib import Path
import tomllib

from scripts.camera_workload_config import _set_table_values


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
CAMERAS = ('/World/PerceptionCamera', '/World/PerceptionCameraSide')
FIXTURE_ROOT = '/World/ReactiveEvaluation'
START = (.45, -.25, .35)
GOAL = (.45, .25, .35)
ORIENTATION = (0., 0., 1., 0.)
SCENES = ('empty', 'static', 'sparse', 'partial', 'narrow', 'large', 'v43')
DIAGNOSTIC_SCENES = ('verified_occlusion',)
SOURCES = ('static', 'frozen_rgbd')
WORKSPACE_MIN = (-.3, -.6, -.02)
WORKSPACE_MAX = (1., .6, 1.12)
MAPPING_MIN = (-.3, -.6, -.02)
MAPPING_MAX = (.85, .6, 1.12)

# This declaration is serialized into the research run before native trials.
# A changed camera/fixture/sampling condition must receive a separate label.
PROTOCOL = dict(
    schema_version=1,
    experiment='global_rgbd_planning_v5',
    source_conditions=SOURCES,
    scenes=SCENES,
    scene=str(SCENE.relative_to(ROOT)),
    baseline_profile=str(PROFILE.relative_to(ROOT)),
    camera_ids=CAMERAS,
    camera_resolution=(640, 480),
    camera_layout='unchanged authored scene; no set_viewpoint or camera repositioning',
    semantic_hz=5., mapping_hz=2.,
    mapping_stride=8, shared_mapping_ray_budget=1024,
    voxel_size_m=.01, evidence_ttl_seconds=5.,
    freshness_limit_seconds=1., automatic_robot_filter=True,
    workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX,
    mapping_min=MAPPING_MIN, mapping_max=MAPPING_MAX,
    motion_strategy='global', collision_model='enhanced', collision_spheres=102,
    rrt_seed=123456, rrt_max_iterations=10000,
    primary_cuboid_budget=64, secondary_diagnostic_cuboid_budget=128,
    secondary_budget_policy='Only diagnose primary evidence omissions; keep as a separately labeled condition',
    start=START, default_target=GOAL, v43_target=START, orientation_wxyz=ORIENTATION,
    position_tolerance_m=.025, orientation_tolerance_rad=.10,
    fixture_source='Unchanged diagnose_reactive_motion fixtures plus the frozen V4.3 main cuboid',
    sparse_fixture_qualification='ThinPost is the historical rectangular box, not a cylinder',
    frozen_world_policy='One accepted worker packet per request; zero active-world refreshes',
    uncertainty_policy='UNKNOWN and OUT_OF_BOUNDS are distinct from verified FREE; no occupancy completion or padding',
    physical_validation='Independent cooked-geometry evaluation may veto execution; it cannot alter the RGB-D planning world',
)


def fixtures(scene):
    """Fresh numeric declarations with stable historical paths and geometry.

    Nothing here queries or supplies geometry to the RGB-D planning source.
    These declarations create physical test fixtures and label post-hoc scores.
    """
    allowed = SCENES+DIAGNOSTIC_SCENES
    if scene not in allowed:
        raise ValueError(f'Unknown V5 scene {scene!r}; expected one of {allowed}')
    values = []
    if scene == 'sparse':
        values = [('ThinPost', (.45,0.,.35), (.035,.06,.20))]
    elif scene in ('static', 'partial', 'verified_occlusion'):
        values = [('Post', (.45,0.,.35), (.09,.12,.26))]
        if scene == 'partial':
            values.append(('Occluder', (.79,-.30,.53), (.08,.40,.40)))
        elif scene == 'verified_occlusion':
            # Separate secondary geometry; never replace the primary partial
            # condition, whose actual camera occlusion must remain measured.
            values.append(('Occluder', (.79,-.015,.40), (.08,.40,.40)))
    elif scene == 'narrow':
        values = [(label, (.45+sign*(.22+.10)/2,0.,.35), (.10,.14,.26))
                  for sign,label in ((-1,'Left'),(1,'Right'))]
    elif scene == 'large':
        values = [('Wall', (.55,.05,.35), (.50,.12,.60))]
    elif scene == 'v43':
        values = [('GlobalRouteDetourV43', (.55,-.225,.30000000000000004), (.06,.06,.12))]
    return [dict(path=f'{FIXTURE_ROOT}/{name}', center=center, dimensions=dimensions)
            for name,center,dimensions in values]


def write_profile(path, source, budget=64):
    """Write one explicit destination TOML file; never overwrite the baseline.

    Both source conditions retain the same RGB-D observation/mapping workload.
    Only global_obstacle_source changes between matched primary profiles. The
    128 bound is available for a separately labeled diagnostic, never implicit.
    """
    if source not in SOURCES:
        raise ValueError(f'Unknown V5 obstacle source {source!r}; expected one of {SOURCES}')
    if type(budget) is not int or budget not in (64,128):
        raise ValueError('V5 permits only its primary 64 or diagnostic 128 cuboid budget')
    path = Path(path)
    if path.resolve() in (SCENE.resolve(), PROFILE.resolve()):
        raise ValueError('Generated profiles must not overwrite the shipped scene/profile')
    text = PROFILE.read_text()
    original = tomllib.loads(text)
    if (original['world']['provider'] != 'multi_view'
            or tuple(original['multi_view']['camera_prim_paths']) != CAMERAS
            or tuple(original['multi_view']['resolution']) != (640,480)):
        raise ValueError('Authored profile no longer matches the declared two-camera baseline')
    text = _set_table_values(text, 'world', dict(update_hz=5.))
    text = _set_table_values(text, 'scene', dict(
        workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX))
    text = _set_table_values(text, 'motion', dict(strategy='global',
        global_collision_model='enhanced', global_obstacle_source=source,
        max_obstacles=budget, rrt_seed=123456, rrt_max_iterations=10000))
    text = _set_table_values(text, 'spatial_mapping', dict(enabled=True,
        source='rgbd', bounds_min=MAPPING_MIN, bounds_max=MAPPING_MAX, voxel_size=.01,
        camera_prim_paths=CAMERAS, stride=8, max_rays=1024,
        automatic_robot_filter=True, update_hz=2., max_frame_age=1.,
        stale_after=1., evidence_ttl=5.))
    text = '# Generated V5 source comparison; authored cameras and defaults remain unchanged.\n'+text
    tomllib.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path

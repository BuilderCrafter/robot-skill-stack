# V5 development-line realignment and publication

14 November 2026 (Europe/Budapest). This task reconciles and publishes existing
work; it introduces no planning feature, experiment, dependency or runtime change.

## Starting state and remote verification

Repository: `https://github.com/BuilderCrafter/robot-skill-stack`.
Host checkout: `/home/ubuntu/projects/robot-skill-stack`.

- Active branch was `finish/cloud-final`, HEAD
  `e11a1fd2635d6b06a0de46feab07fb7b20cd4550`.
- Local `feature/global-rgbd-planning-v5` and its remote-tracking ref had the
  same HEAD. `git fetch origin` succeeded; an independent `git ls-remote --heads
  origin refs/heads/feature/global-rgbd-planning-v5` confirmed the same SHA.
- Neither local branch had any commits beyond the fetched V5 baseline.
  No commits needed cherry-picking, merging or rebasing.
- Staged: new `AGENTS.md`, 158 lines. Unstaged: candidate changes in
  `robot_skill_stack/integrations/isaac/manipulation/motion/collision_model.py`
  (73 insertions, two deletions).
- Untracked: the familiarization report, bounded-improvement report, and
  `tests/unit/test_rgbd_base_clearance.py` (nine tests).

All branches, staged/unstaged differences, untracked files, instructions,
reports and candidate tests were inspected before reconciliation. V4.3 remains
ancestor `aef5f2fe0af0229a1c69c64035b7de0b07173b0b`; the fetched V5 checkpoint
already contains that work. `finish/cloud-final` is retained locally unchanged.

## Recovery checkpoint

Before switching branches or selectively transferring files:

1. Recorded initial status, index entries, original file SHA256 values and a
   binary tracked patch under `.git/recovery/v5-realignment-20261114/`.
2. Created local branch `recovery/v5-realignment-20261114-pre-transfer` at the
   original HEAD.
3. Ran `git stash push --include-untracked --message
   'recovery: complete pre-realignment cloud state 2026-11-14'`.
4. Pinned the stash with permanent local ref
   `refs/recovery/v5-realignment-20261114-worktree`.
5. Read back and SHA256-verified every saved file, using the stash's worktree,
   index and untracked trees. Separately verified the exact staged `AGENTS.md`.

| Recovery object | Exact SHA |
| --- | --- |
| Worktree/stash merge commit | `edc1b467c6bf03c192d0d9dccce2b538309a024f` |
| Original staged index | `e396471e9fa16b1d825dd3bb37d50a9689bef072` |
| Original untracked files | `38d5ebce4ec0c4fec57354f115574d08b37e9549` |
| Original HEAD / recovery branch | `e11a1fd2635d6b06a0de46feab07fb7b20cd4550` |

Both `refs/stash` and the explicitly pinned recovery ref are retained; no stash
was popped or dropped. Recovery objects are local only and are not part of the
published branch history. Ignored outputs were not included in the stash and
remain in place; no cleanup command was used.

To inspect the excluded candidate without changing the active worktree:

```bash
git show refs/recovery/v5-realignment-20261114-worktree:robot_skill_stack/integrations/isaac/manipulation/motion/collision_model.py
git show refs/recovery/v5-realignment-20261114-worktree^3:tests/unit/test_rgbd_base_clearance.py
```

To recover the entire original staged/unstaged/untracked state in a separate,
new worktree (choose an unused destination):

```bash
git worktree add --detach /tmp/robot-v5-pre-realignment \
  recovery/v5-realignment-20261114-pre-transfer
git -C /tmp/robot-v5-pre-realignment stash apply --index \
  refs/recovery/v5-realignment-20261114-worktree
```

The snapshot verification was performed directly against Git objects; these
optional recovery commands were not executed during reconciliation.

## Classification and transfer decision

| Class | Discovered work | Disposition |
| --- | --- | --- |
| Verified useful production change | None established by the bounded candidate | No new runtime behavior published |
| Inadequately validated / stopped implementation | RGB-D-only enclosing-box separation helper and supplementary base-preflight branch | Preserved exactly in local recovery; excluded from V5 |
| Candidate-only tests | Nine methods testing separation, tangency, transforms, numerical uncertainty, source gating and plane preservation | Preserved in recovery with their implementation; excluded from active test discovery |
| Experimental findings | `reports/finalization/rgbd-bounded-improvement.md` | Retained; status note identifies candidate as excluded and historical worktree section as a snapshot |
| Reconnaissance | `reports/finalization/project-familiarization.md` | Retained as a dated source/evidence audit; status note points here for the publication decision |
| Unrelated pre-existing staged work | `AGENTS.md` | Valid project instructions, no credentials/unrelated content; preserved byte-for-byte in a separate commit |
| Generated/ignored material | Native JSON/logs, camera and occupancy arrays, cooked geometry, traces, task-local scripts, private stream environment, `.deps`, caches | Left in place outside Git; no output directory, environment or credential file staged |
| This reconciliation | This handover | Published with validation and recovery details |

After stashing, switched to the existing tracking branch with
`git switch feature/global-rgbd-planning-v5`. Restored only `AGENTS.md` and the
two reports from their verified snapshot blobs. The candidate source/test were
not reapplied, so production behavior returned to the original V5 base preflight.
No wholesale merge from `finish/cloud-final`, reset, destructive checkout,
published-history rewrite, main merge or branch deletion occurred.

## Assessment of the bounded RGB-D candidate

The prior code did implement a more precise separation test using the existing
source-bound base AABB. A positive projected enclosure gap, less the 10 µm
frame allowance and numerical guard, conditionally proves separation. Overlap,
tangency, malformed geometry and ambiguous bounds reject. Source tags confined
the supplement to exact RGB-D cuboids, preserving GT/static and known-support
paths and all occupancy evidence. The mathematical idea is defensible within
the existing geometric/frame assumptions, but it was not a generally certified
physical safety improvement.

Prior offline validation passed 1,382 discovered tests (1,373 original plus
nine candidate tests), architecture/syntax and eight standalone modules; the
prior container run passed 22 focused cases under Isaac Python 3.11. Those
results support the implementation's tested invariants, not its native utility.

Existing ignored artifacts were independently reread in this task:

| Prior measurement | Actual outcome |
| --- | --- |
| Fresh V5 `v43/static`, budget 64 | Exit 0, measurement PASS, selected GT plan and movement success; 3 waypoints, 78 actions, zero reports/physical-contact steps, 15.922 mm sampled box clearance, 1.060 mm / 0.001700 rad final error; evaluator allows 286/286 sampled configurations |
| Fresh `static/frozen_rgbd`, budget 64 | Exit 0, measurement PASS, source rejected at 1.060005840 s age against 1.000 s limit; no plan, no physical trajectory evaluation, zero selected-request actions or movement |
| Offline audit of that rejected-source packet | `source_usable=false`; 64 selected cuboids, one sphere rejection, one supplemented rejection, **zero resolved rejections** |
| Partial-scene regression | Not run after the primary stop condition; candidate veto preservation remains unverified |
| Unsafe RGB-D trajectories newly returned in these runs | None: the selected RGB-D request never reached planning; historical unsafe V5 routes are unchanged evidence |
| Research-level gain / obstacle-induced RGB-D detour | Neither demonstrated |
| Previous stream restoration | Receipt confirms original argv/configuration, app ready, streaming app loaded and TCP 49100 listener; client video was not observed |

The unresolved archived cell has center `(0.025,0.095,-0.005)` m and 1 cm sides.
Its best box projection gap is only about 24 nm; after the unchanged 10 µm
allowance it still cannot prove clearance. This supports the declared
**no-useful-distinction stop condition for that packet**. It does not establish
that all possible AABB refinements are useless, or that the source-rejected
native trial exercised the candidate geometry code.

Decision: publish the negative findings, restore original V5 production behavior,
and retain the candidate only in local recovery. No new experiment or second
solution was attempted. The research-only physical veto is not represented as
a production safeguard.

### Local evidence identities

The files below remain ignored and are **not** available from the GitHub report
alone. SHA256 values bind this review to the actual local evidence, not an
inferred process success. Raw data were neither overwritten nor reclassified.

| Local artifact | SHA256 |
| --- | --- |
| `outputs/motion-planning-v5/rgbd-base-clearance-20261113/v43/static/result.json` | `9d3f8c4c61b8a472f2940605d3efb21ec1e75edab259f700039151a7d24ee005` |
| `outputs/motion-planning-v5/rgbd-base-clearance-20261113/static/frozen_rgbd/result.json` | `99306515908489a0ae9bfa1fe8d10d990c2ec87712ce048a01fc2de86b7e1e6d` |
| `outputs/rgbd-base-clearance-20261113/packet-audit.json` | `7cf371774876b794e1d372e44ee97c88bc5cd63cfd05834441fb52fc84b2a28f` |
| `outputs/rgbd-base-clearance-20261113/stream-restored.json` | `2cfef6b3d3e9ee1458f3b97b755c24debf4d306af637d97dd209221ae2a2cb5c` |

## Validation of the reconciled tree

The old `/tmp/robot-recon-py312` environment no longer existed. Recreated an
isolated `/tmp/robot-v5-realignment-py312` using CPython 3.12.16 with NumPy
1.26.4, py_trees 2.6.0, SciPy 1.15.3 and PyYAML 6.0.2. Repository requirements,
`.deps`, Isaac installation and workstation launchers were unchanged.

```bash
uv venv --python /home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12 \
  /tmp/robot-v5-realignment-py312
uv pip install --python /tmp/robot-v5-realignment-py312/bin/python \
  numpy==1.26.4 py_trees==2.6.0 scipy==1.15.3 PyYAML==6.0.2

/tmp/robot-v5-realignment-py312/bin/python -m unittest \
  tests.unit.test_global_collision_model_v4_2 \
  tests.unit.test_franka_collision_model_v4_2 \
  tests.unit.test_global_frozen_rgbd_v5 tests.unit.test_lula_global
# PASS: 52 tests in 1.328 s.

/tmp/robot-v5-realignment-py312/bin/python scripts/run_all_regressions.py \
  --unit-only --artifacts-dir /tmp/robot-v5-realignment-unit-20261114
```

Full runner: **PASS**, 1,373 discovered tests in 104.539 s, architecture/syntax
PASS and all eight standalone regression modules PASS (all ten runner stages,
overall exit 0). Retained logs and summary:
`/tmp/robot-v5-realignment-unit-20261114/run-pbj5cc7k/summary.json` and neighboring
`unit_discovery.log`. No native validation was needed for documentation-only
publication with production restored byte-for-byte.

Expected test-count change is 1,382 → **1,373**, solely because the nine
excluded-candidate tests are no longer in discovery. No original test, fixture
expectation, safety threshold or source hash was edited to obtain a pass.

`git diff e11a1fd -- robot_skill_stack tests config scripts run_isaac_python.sh
run_isaac_sim.sh` is empty. The reconciled production, test, resource and launcher
trees equal the original V5 baseline. Only instructions and reports are intended
for publication. Document links, new-file whitespace and explicit staged-file
review are checked before committing.

## Environment status and native boundary

No native Isaac process was launched, stopped or restarted in this task.
Read-only `docker ps -a --filter name=isaac51` found the existing container
already **Exited (137), approximately 11 hours before inspection**. Therefore
the stream was not operational on arrival; the cause is not established here.
The previous task's successful server restoration remains a historical fact,
not current uptime evidence. No current client-video verification is claimed.

The container stayed stopped throughout this reconciliation. VM, Docker,
network, drivers, Tailscale/NetBird and firewall configuration were not changed.
Restoring the current stream is a separate operational issue; it is not needed
to publish documentation-only changes and was not attempted.

## Publication and next boundary

Active development branch: **`feature/global-rgbd-planning-v5`**, tracking
`origin/feature/global-rgbd-planning-v5`. Publication uses explicit paths and a
normal fast-forward push to that branch only. `main`, `finish/cloud-final`, and
the local recovery refs are not pushed or merged.

| Commit | Content |
| --- | --- |
| [`ec9a110ec653ebff0a767eb43467ba6d50c39896`](https://github.com/BuilderCrafter/robot-skill-stack/commit/ec9a110ec653ebff0a767eb43467ba6d50c39896) | Byte-identical pre-existing project instructions |
| [`61c49cafd9e61a4bfd145538472dfc30dbcd6433`](https://github.com/BuilderCrafter/robot-skill-stack/commit/61c49cafd9e61a4bfd145538472dfc30dbcd6433) | Familiarization and stopped-experiment reports with publication status notes |
| This report's commit, parent `61c49cafd9e61a4bfd145538472dfc30dbcd6433`, subject `docs: record V5 realignment and validation` | Reconciliation, recovery, test results and publication handover |

**Final pushed remote HEAD is the commit introducing this report**, identified
uniquely by its parent and subject above. Its exact SHA is included in the task
handoff and `.git/recovery/v5-realignment-20261114/publication.json`; embedding
a commit's own SHA inside its file would change that SHA. It is also available
in [the report's GitHub commit history](https://github.com/BuilderCrafter/robot-skill-stack/commits/feature/global-rgbd-planning-v5/reports/finalization/v5-realignment.md).

Publication verification, executed after creating that commit:

```bash
git push origin HEAD:refs/heads/feature/global-rgbd-planning-v5
git ls-remote --exit-code origin refs/heads/feature/global-rgbd-planning-v5
git fetch origin feature/global-rgbd-planning-v5
git rev-parse HEAD origin/feature/global-rgbd-planning-v5
git merge-base --is-ancestor e11a1fd2635d6b06a0de46feab07fb7b20cd4550 HEAD
git merge-base --is-ancestor aef5f2fe0af0229a1c69c64035b7de0b07173b0b HEAD
git status --porcelain=v1 --untracked-files=all
```

Acceptance requires the independently advertised remote SHA to equal local
HEAD, both publication-parent commits and the original V5/V4.3 commits to remain
ancestors, a clean worktree, and a branch diff containing **only `AGENTS.md`
and these three report files**. The local publication receipt records the exact
push/verification result and commit list; no recovery or ignored artifact is
part of that push. No automatic PR is created.

Remaining issues: strict V4.2/V4.3 archival replay inputs/path portability;
experimental RGB-D evidence/freshness/geometry limitations; a visible project
GUI/GT demonstration workflow; stale current-feature documentation; and the
already-stopped container/stream. No candidate safety behavior is awaiting
acceptance on this published branch because it was excluded.

Stop after publication and independent remote-HEAD verification. No PR, merge,
new RGB-D work, GUI refactor, hybrid or presentation implementation is part of
this task.

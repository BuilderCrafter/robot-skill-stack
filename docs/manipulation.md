# Manipulation

Home, MoveToPose, Pick and Place share one command path:

```python
await runtime.execute_async("pick", object_id="object_1")
await runtime.execute_async("place", target=target, mode="stable")
await runtime.execute_async("home")
```

`RobotRuntime(registry, world_model)` shares the skills' cached model. A skill
implements `execute_async` and declares `name`; registration prevents duplicate
names. Results contain status (`success`, `failed`, `timeout`), failure code,
message and diagnostic details. Busy commands are rejected, not queued.
Backend queries are synchronous; commands yield on Kit's simulator thread.

## Pick, grasp planning and feasibility

`SimplePrimitiveGraspPlanner` produces conservative top-down grasps for
cubes/boxes, spheres and upright/lying cylinders. It requires known supported
geometry, pose and graspability; checks parallel-jaw fit, contact/support-plane
and palm clearance, known-object swept envelopes and endpoint IK. Unknown,
ambiguous or unsupported tilted geometry is rejected. A future grasp planner
can implement `GraspPlanner`; no learned/arbitrary-object planner is implemented.

Franka's adapter-owned usable width is 75 mm: a 50 mm cube is accepted and a
90 mm cube rejected before motion. Pick compares current visible position with
the planned grasp after approach. Movement above 20 mm triggers a local replan,
limited to two retries before `OBJECT_MOVED`.

`SimplePlacementPlanner` rejects occupied targets before any backend motion or
gripper command, using conservative AABB overlap with visible known objects.
Invisible/stale objects are not hard obstacles. Reachability checks test
endpoints, not full paths. Neither this planner nor IK establishes collision-
free transport; scene obstacles/support plane are not registered with RMPflow.

## Place and intentional release

The target is the **object center**, transformed to a TCP target using the
measured attachment offset. The physical sequence remains approach, descent,
verified opening, retreat. There are no automatic retreat retries or recovery
commands after a failure.

- `stable` (default and the GUI's only mode) requires a visible final object
  pose within the skill's 50 mm tolerance. The native benchmark independently
  retains its stricter 25 mm physical-error gate.
- `release` intentionally allows a drop from the requested release pose without
  landing position/visibility verification. It still requires reachability,
  occupancy checks, successful approach/opening and successful retreat. There
  is no GUI drop toggle. The legacy `default` mode alias still means `stable`.

After opening/retreat, details distinguish:

| Field | Meaning |
| --- | --- |
| `release_confirmed` | Opening succeeded and position evidence confirmed Free; false for a failed opening, even if subsequent feedback establishes Free |
| `placement_verified` | Stable-mode position check passed/failed; `None` when not checked, including intentional release |
| `retreat_ok` | Retreat succeeded/failed; `None` if opening failed and retreat was not attempted |
| `failed_stage` | `release`, `verification` or `retreat` on these failures |
| `retreat_message`, `retreat_details` | Backend evidence for the attempted retreat |
| `verification_message`, `placement_error_m` | Verification failure explanation / measured error when available |

A failed retreat now returns FAILED/`PLACE_FAILED`, or TIMEOUT/`TIMEOUT` for a
backend timeout, in both modes. If stable verification also failed, retreat
failure takes precedence while verification evidence remains in details.
Confirmed release stays Free with no attachment; retreat failure never marks
it Held again. Failed opening stops before retreat; measured gripper evidence
determines Free/Held/Uncertain, reported as `gripper_state`. A supported opening
timeout preserves TIMEOUT (uncertainty retains the `GRASP_UNCERTAIN` code).
Pre-release motion failures remain failures. Exceptions/cancellation retain the
existing runtime/held-state handling; these detail fields describe returned
backend outcomes, not a persistent transaction journal. Callers must inspect the
WorldModel after interruption and must not blindly retry a released object.

## Home definition

For Franka, `home` is a **copy of the measured startup TCP pose**, captured when
the backend is constructed after bootstrap reset and startup updates. It is not
a specified joint configuration or a universally safe pose. `home_async(name)`
selects a named pose and uses the existing Cartesian controller/tolerances;
unknown names issue no command. The GUI label and BT API remain `Home`/`home`.

Bootstrap resets SDK scene defaults, but this repository specifies no separately
validated collision-free Home joint state. SDK reset support alone does not
make a return path safe, and a TCP goal can use a different joint configuration.
Phase 4C preserves the working target/trajectory, corrects the description and
adds a tooltip rather than inventing angles or coordinates. A new Home target
requires scene-specific native validation. Uncertain state blocks Home; carrying
Home uses the same loss guard as other motions. Recovery Home is subject to the
existing pre-closure Pick policy, not an emergency recovery guarantee.

## Held state and operator recovery

Finger positions estimate attachment, without force sensing:

| State | Meaning and consequence |
| --- | --- |
| Free | No software attachment after empty/open evidence or explicit operator reset |
| Held | Position evidence supports the candidate grasp; ID and measured TCP offset are retained |
| Uncertain | Candidate may still be held; ID/offset remain protected and automatic manipulation is blocked |

Pick reserves the candidate as Uncertain before closing and confirms Held before
lift. Empty closure clears attachment; unresolved feedback retains Uncertain.
Verified release clears attachment but keeps its short placement association
hint. Failed/interrupted opening clears the hint; attachment follows feedback.

Franka uses named finger DOF `lower`/`upper` values from `robot.dof_properties`,
not the SDK's nominal opening target or unavailable `get_dof_limits()` method.
Measured 40 mm travel can be fully open even if the SDK asks for 50 mm per finger.
Finite readings within physical travel, intermediate width under a close
command and usable aperture support Held; empty closure below 5 mm or full
opening supports Free. Invalid/partial/ambiguous readings are Uncertain.
Stationary inspection requires three consistent known-state samples, width
changes at most 1 mm and advancing simulation time, with at most 12 updates.
Existing open/close durations remain 60/90 updates.

While carrying, three consecutive suspect samples stop motion through the
existing arm hold. Confirmed loss clears attachment/hints, marks the cached
object invisible until a fresh observation, and reports `GRASP_LOST`. Suspect
feedback at completion is resolved before success. Exceptions/cancellation
attempt bounded stationary resolution while preserving the original error.
A failed operation never automatically resumes just because evidence resolves.

Unresolved uncertainty requires operator inspection. Confirmed **Clear Held**
calls `runtime.clear_held_state(expected_id)` when idle, clearing software state
only; it cannot release a physical object. The operator must first establish
that clearing is appropriate. A changed ID or busy runtime rejects the reset.

Finger width cannot establish identity, distinguish every obstruction, prevent
a drop or detect every slip before motion ends. Position evidence is delayed;
arm holding is not an emergency stop and physical settling can continue.
Controller and native validation limits are in [diagnostics](control_diagnostic.md)
and [validation](validation.md).

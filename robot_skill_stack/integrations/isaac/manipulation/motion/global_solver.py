"""Temporary TGS accuracy for global trajectories; never edit saved assets."""
from __future__ import annotations

from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


POSITION_ATTRIBUTE = "physxArticulation:solverPositionIterationCount"
TGS_POSITION_ITERATIONS = 255


def _nondefault_info(spec):
    return {key: repr(spec.GetInfo(key)) for key in spec.ListInfoKeys() if key != "default"}


class PositionIterationLease:
    """Own one session-layer default, preserving foreign opinions on cleanup.

    All accesses use the captured stage, including after Stop/reset/rebuild.
    USD-backed counts are requested values, not native island-count readbacks.
    """
    def __init__(self, stage, prim_path, requested):
        from pxr import Usd
        self.stage, self.session, self.path = stage, stage.GetSessionLayer(), str(prim_path)
        self.closed = False
        self._activation_completed = False
        self.attribute = stage.GetPrimAtPath(self.path).GetAttribute(POSITION_ATTRIBUTE)
        self.before = int(self.attribute.Get())
        self.requested = max(self.before, int(requested))
        self.changed = self.before != self.requested
        self.restored = False
        self.opinion = self.session.GetAttributeAtPath(self.path+"."+POSITION_ATTRIBUTE)
        self.had_opinion = self.opinion is not None
        self.had_default = self.had_opinion and self.opinion.HasInfo("default")
        self.default = self.opinion.GetInfo("default") if self.had_default else None
        parts = self.path.strip("/").split("/")
        self.owned_ancestors = ["/"+"/".join(parts[:i]) for i in range(len(parts), 0, -1)
            if self.session.GetPrimAtPath("/"+"/".join(parts[:i])) is None]
        if self.changed:
            try:
                with Usd.EditContext(self.stage, self.session):
                    if not self.attribute.Set(self.requested) or int(self.attribute.Get()) != self.requested:
                        raise RuntimeError("Requested TGS position iteration count was not applied")
                self._created_info = _nondefault_info(
                    self.session.GetAttributeAtPath(self.path+"."+POSITION_ATTRIBUTE))
                self._activation_completed = True
            except BaseException as error:
                try:
                    self.close()
                except Exception as cleanup:
                    error.add_note(f"Solver opinion cleanup also failed: {cleanup}")
                raise

    @property
    def diagnostics(self):
        return dict(solver_type="TGS", requested_position_iterations_before=self.before,
            requested_position_iterations_during=self.requested, changed=self.changed,
            original_session_opinion_restored=self.restored,
            scope="Temporary articulation session opinion; no velocity/gain/limit or saved-file edits",
            count_provenance="USD requested count; native effective island count is not read here")

    def close(self):
        if self.closed:
            return
        from pxr import Sdf, Usd
        if self.changed:
            current = self.session.GetAttributeAtPath(self.path+"."+POSITION_ATTRIBUTE)
            present = current is not None and current.HasInfo("default")
            original = ((current is not None) == self.had_opinion and present == self.had_default
                and (not present or current.GetInfo("default") == self.default)
                and int(self.attribute.Get()) == self.before)
            if original:
                self.restored, self.closed = True, True
                return
            if (self._activation_completed and
                    (not present or current.GetInfo("default") != self.requested)):
                raise RuntimeError("Solver session default changed by another owner; refusing to overwrite it")
            if (self._activation_completed and not self.had_opinion
                    and _nondefault_info(current) != self._created_info):
                raise RuntimeError("Solver session metadata changed by another owner; refusing to remove it")
            with Usd.EditContext(self.stage, self.session):
                if not self.had_opinion:
                    self.stage.GetPrimAtPath(self.path).RemoveProperty(POSITION_ATTRIBUTE)
                elif self.had_default:
                    self.attribute.Set(self.default)
                else:
                    # Clear only our default; preserve time samples/metadata.
                    self.attribute.ClearAtTime(Usd.TimeCode.Default())
                for path in self.owned_ancestors:
                    spec = self.session.GetPrimAtPath(path)
                    if (spec is not None and spec.specifier == Sdf.SpecifierOver
                            and not list(spec.properties) and not list(spec.nameChildren)
                            and set(spec.ListInfoKeys()).issubset({"specifier"})):
                        self.stage.RemovePrim(path)
            opinion = self.session.GetAttributeAtPath(self.path+"."+POSITION_ATTRIBUTE)
            present = opinion is not None and opinion.HasInfo("default")
            if ((opinion is not None) != self.had_opinion or present != self.had_default
                    or present and opinion.GetInfo("default") != self.default
                    or int(self.attribute.Get()) != self.before):
                raise RuntimeError("Original solver session opinion was not restored")
        self.restored, self.closed = True, True


class _UnchangedSolver:
    diagnostics = dict(solver_type="PGS", changed=False,
        scope="TGS compatibility setting does not apply to PGS")

    def close(self):
        pass


def execution_solver_scope(backend):
    """Acquire only after a validated moving plan, before its first command."""
    try:
        context = backend.world.get_physics_context()
        solver = context.get_solver_type()
        if solver == "PGS":
            return _UnchangedSolver()
        if solver != "TGS":
            raise RuntimeError(f"Unsupported solver type: {solver!r}")
        import omni.usd
        from pxr import PhysxSchema
        stage = omni.usd.get_context().get_stage()
        scene = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(context.prim_path))
        maximum = int(scene.GetMaxPositionIterationCountAttr().Get())
        if maximum < TGS_POSITION_ITERATIONS:
            raise RuntimeError("Physics-scene maximum does not permit validated TGS accuracy (255)")
        return PositionIterationLease(stage, backend.robot.prim_path, TGS_POSITION_ITERATIONS)
    except Exception as error:
        raise PlanningFailure("solver_configuration_failure",
            f"Global execution solver configuration failed: {error}") from error

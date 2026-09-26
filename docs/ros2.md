# ROS 2 integration boundary

ROS 2 is not implemented in the current course-level stack.

If the course requires a ROS pipeline, the preferred design is a thin adapter under `robot_skill_stack/integrations/ros2/` rather than rewriting the internal architecture around ROS.

The adapter would translate between ROS messages/actions and the two stable internal boundaries:

- `RobotRuntime`: execute `pick`, `place`, `move_to_pose`, `home`, and return structured `SkillResult`s.
- `WorldModel`: publish/query the current object state.

A likely minimal course adapter would expose long-running manipulation through ROS 2 Actions and publish WorldModel snapshots on a topic. Perception, grasp planning, skills, and the Isaac backend would remain unchanged.

Before implementing this, clarify with the professor whether “ROS pipeline” means only an external interface or whether ROS/MoveIt is expected to sit inside the control path.

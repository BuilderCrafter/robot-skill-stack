"""Keep windows alive until an update after the click/draw that closed them."""
import asyncio
import carb
import omni.kit.app

_PENDING = set()


def retire_window(window):
    if window is None:
        return
    window.set_visibility_changed_fn(lambda visible: None)
    window.visible = False

    async def dispose():
        await omni.kit.app.get_app().next_update_async()
        # One further turn also separates a close originating in an update event.
        await omni.kit.app.get_app().next_update_async()
        window.destroy()

    task = asyncio.ensure_future(dispose())
    _PENDING.add(task)

    def done(completed):
        _PENDING.discard(completed)
        if not completed.cancelled() and completed.exception() is not None:
            carb.log_error(f'[robot_skill_stack.ui] deferred window cleanup: {completed.exception()}')
    task.add_done_callback(done)

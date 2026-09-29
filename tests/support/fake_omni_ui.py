"""Recording widgets for callback/layout tests. This is NOT an OmniUI renderer."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'exts' / 'robot_skill_stack.ui'


class ValueModel:
    def __init__(self, value=0, changed=None):
        self.value, self.changed = value, changed

    def set_value(self, value):
        self.value = value
        if self.changed:
            self.changed()

    def get_value_as_string(self): return str(self.value)
    def get_value_as_float(self): return float(self.value)
    @property
    def as_int(self): return int(self.value)


class ComboModel:
    def __init__(self, index):
        self.callbacks = []
        self.value = ValueModel(index, self._changed)

    def _changed(self):
        for callback in self.callbacks:
            callback(self, None)

    def add_item_changed_fn(self, callback): self.callbacks.append(callback)
    def get_item_value_model(self): return self.value


class Length(float):
    """Test-only numeric storage; native widget setters require a Length object."""


class Pixel(Length):
    pass


class Fraction(Length):
    pass


class RecordingUI(ModuleType):
    def __init__(self, *, defer_builds=False):
        super().__init__('omni.ui')
        self.in_event = False
        self.defer_builds = defer_builds
        self.nodes, self.stack = [], []
        self.Alignment = SimpleNamespace(LEFT_CENTER=0, RIGHT_CENTER=1, CENTER=2)
        self.ScrollBarPolicy = SimpleNamespace(SCROLLBAR_ALWAYS_OFF=0, SCROLLBAR_AS_NEEDED=1)
        self.FillPolicy = SimpleNamespace(PRESERVE_ASPECT_FIT=0)
        self.Direction = SimpleNamespace(TOP_TO_BOTTOM=0, LEFT_TO_RIGHT=1)
        self.Length, self.Pixel, self.Fraction = Length, Pixel, Fraction
        for kind in ('Window', 'Frame', 'HStack', 'VStack', 'ZStack', 'ScrollingFrame', 'Spacer',
                     'Label', 'Rectangle', 'Button', 'Circle', 'Image', 'Separator', 'FloatField',
                     'StringField', 'ComboBox'):
            setattr(self, kind, self._factory(kind))

    def _factory(self, kind):
        def create(*args, **kwargs):
            return Widget(self, kind, args, kwargs)
        return create

    def find(self, kind=None, **properties):
        return [node for node in self.nodes if not node.destroyed and (kind is None or node.kind == kind)
                and all(getattr(node, k, None) == v for k, v in properties.items())]

    def draw_cycle(self):
        """Model deferred builds and zero-height rejection, NOT pixel layout or rendering."""
        built = 0
        for node in tuple(self.nodes):
            if node.destroyed or not node._pending_build:
                continue
            ancestor, visible = node, True
            while ancestor is not None:
                if not ancestor.visible or (ancestor.kind == 'Frame' and getattr(ancestor, 'height', None) == 0):
                    visible = False
                    break
                ancestor = ancestor.parent
            if visible:
                node._build()
                built += 1
        return built


class Widget:
    def __init__(self, ui, kind, args, kwargs):
        self.ui, self.kind, self.args = ui, kind, args
        self.children, self.rebuilds = [], 0
        self.destroyed = self._pending_build = False
        self._visible, self.visibility_callback = True, None
        self.enabled, self.style, self.tooltip, self.name = True, {}, '', ''
        self.scroll_y = 0.
        self.text = args[0] if args else ''
        self.model = ComboModel(args[0]) if kind == 'ComboBox' else ValueModel()
        self.parent = ui.stack[-1] if ui.stack and kind != 'Window' else None
        if self.parent:
            self.parent.children.append(self)
        ui.nodes.append(self)
        # Constructor keywords accept scalars; later property assignments do not.
        for name, value in kwargs.items():
            if name in ('height', 'width') and kind != 'Window' and not isinstance(value, Length):
                value = Pixel(value)
            setattr(self, name, value)
        self.build_fn = kwargs.get('build_fn')
        if kind == 'Window':
            with self:
                self.frame = ui.Frame()
        if self.build_fn:
            self.rebuild()

    def __setattr__(self, name, value):
        if (name in ('height', 'width') and getattr(self, 'kind', 'Window') != 'Window'
                and not isinstance(value, Length)):
            raise TypeError(f'{name} setter expects omni.ui.Length; got {type(value).__name__}: {value!r}')
        object.__setattr__(self, name, value)

    def __enter__(self):
        self.ui.stack.append(self)
        return self

    def __exit__(self, *exc):
        assert self.ui.stack.pop() is self

    def rebuild(self):
        self.rebuilds += 1
        self._pending_build = True
        if not self.ui.defer_builds:
            self._build()

    def _build(self):
        self._pending_build = False
        for child in self.children:
            child.destroy()
        self.children = []
        if self.build_fn:
            with self:
                self.build_fn()

    def set_mouse_released_fn(self, callback): self.mouse_released_fn = callback
    def set_visibility_changed_fn(self, callback): self.visibility_callback = callback

    @property
    def visible(self): return self._visible
    @visible.setter
    def visible(self, value):
        self._visible = value
        if self.visibility_callback:
            self.visibility_callback(value)

    def destroy(self):
        if self.kind == 'Window' and self.ui.in_event:
            raise RuntimeError('Container::destroy during a UI event')
        self.destroyed = True
        for child in self.children:
            child.destroy()

    def click(self):
        if self.enabled:
            self.ui.in_event = True
            try:
                self.clicked_fn()
            finally:
                self.ui.in_event = False


@contextmanager
def extension_environment(*, defer_builds=False):
    owned_loop = None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        owned_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(owned_loop)
    ui = RecordingUI(defer_builds=defer_builds)
    omni, ext, kit, app = (ModuleType(n) for n in ('omni', 'omni.ext', 'omni.kit', 'omni.kit.app'))
    omni.ext, omni.kit, omni.ui = ext, kit, ui
    kit.app, ext.IExt = app, object
    stream = SimpleNamespace(create_subscription_to_pop=lambda fn, **kw: SimpleNamespace(callback=fn))
    application = SimpleNamespace(get_update_event_stream=lambda: stream,
                                  next_update_async=lambda: asyncio.sleep(0))
    app.get_app = lambda: application
    carb = ModuleType('carb')
    carb.log_info = carb.log_warn = carb.log_error = lambda message: None
    bootstrap = ModuleType('robot_skill_stack.integrations.isaac.bootstrap')
    async def unavailable(*args):
        raise AssertionError('Isaac build_runtime was not replaced by the test')
    bootstrap.build_runtime = unavailable
    modules = {m.__name__: m for m in (omni, ext, kit, app, ui, carb, bootstrap)}
    saved_path = sys.path[:]
    with patch.dict(sys.modules, modules):
        for name in tuple(sys.modules):
            if name.startswith('robot_skill_stack_ui'):
                del sys.modules[name]
        sys.path.insert(0, str(EXT))
        try:
            module = importlib.import_module('robot_skill_stack_ui.ui.extension')
            controller = module.RobotSkillStackExtension()
            controller.on_startup('test.extension')
            yield ui, controller, module
        finally:
            if owned_loop is not None:
                asyncio.set_event_loop(owned_loop)
            if 'controller' in locals() and not controller._shutting_down:
                controller.on_shutdown()
            if owned_loop is not None:
                owned_loop.run_until_complete(asyncio.sleep(0))
                owned_loop.run_until_complete(asyncio.sleep(0))
                owned_loop.run_until_complete(asyncio.sleep(0))
                owned_loop.close()
                asyncio.set_event_loop(None)
            sys.path[:] = saved_path

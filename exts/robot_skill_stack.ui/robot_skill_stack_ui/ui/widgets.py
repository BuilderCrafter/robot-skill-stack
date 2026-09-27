from contextlib import contextmanager

import omni.ui as ui

from .style import ICONS, badge_style


def image(name, size=20, **kwargs):
    return ui.Image(str(ICONS / f'{name}.png'), width=size, height=size,
                    fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT, **kwargs)


def button(text, callback, *, icon=None, **kwargs):
    if icon:
        kwargs.update(image_url=str(ICONS / f'{icon}.png'), image_width=18, image_height=18, spacing=8)
        style = dict(kwargs.get('style', {}))
        style['Button'] = {**style.get('Button', {}), 'stack_direction': ui.Direction.LEFT_TO_RIGHT}
        style['Button.Image'] = {**style.get('Button.Image', {}), 'alignment': ui.Alignment.CENTER}
        if text:
            style['Button.Label'] = {**style.get('Button.Label', {}), 'alignment': ui.Alignment.LEFT_CENTER}
        kwargs['style'] = style
    kwargs.setdefault('height', 30)
    return ui.Button(text, clicked_fn=callback, **kwargs)


@contextmanager
def padded(padding=9, **kwargs):
    with ui.HStack(**kwargs):
        ui.Spacer(width=padding)
        with ui.VStack(spacing=0):
            ui.Spacer(height=padding)
            yield
            ui.Spacer(height=padding)
        ui.Spacer(width=padding)


@contextmanager
def section(title, icon):
    with ui.ZStack(height=0):
        ui.Rectangle(name='panel')
        with padded():
            with ui.VStack(height=0, spacing=8):
                with ui.HStack(height=18, spacing=9):
                    image(icon, 18)
                    ui.Label(title.upper(), name='section')
                with ui.VStack(height=0, spacing=6):
                    yield


def badge(state):
    rectangle, label = badge_style(state)
    with ui.ZStack(width=66 if state == 'VISIBLE' else 50, height=22):
        ui.Rectangle(style=rectangle)
        ui.Label(state, alignment=ui.Alignment.CENTER, style=label)


def vector(values):
    return '[ --, --, -- ]' if values is None else '[ ' + ', '.join(f'{v:.3f}' for v in values) + ' ]'

from pathlib import Path

ICONS = Path(__file__).resolve().parents[2] / 'icons'


def color(hex_rgb):
    value = hex_rgb.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (255 << 24) | (b << 16) | (g << 8) | r


BG = color('13191f')
PANEL = color('1c232b')
CARD = color('181f27')
BORDER = color('303c49')
TEXT = color('e1eaf6')
MUTED = color('adc1da')
DIM = color('8399b3')
BLUE = color('2855a0')
GREEN = color('60dc98')
RED = color('ef9588')
AMBER = color('e7bd74')

STYLE = {
    'Window': {'background_color': BG},
    'Label': {'color': TEXT, 'font_size': 14},
    'Label::title': {'font_size': 27, 'color': TEXT},
    'Label::world_title': {'font_size': 22, 'color': TEXT},
    'Label::section': {'font_size': 13, 'color': color('a5c9f4')},
    'Label::muted': {'color': MUTED, 'font_size': 13},
    'Label::dim': {'color': DIM, 'font_size': 12},
    'Label::detail': {'color': MUTED, 'font_size': 13},
    'Rectangle::background': {'background_color': BG},
    'Rectangle::panel': {'background_color': PANEL, 'border_color': BORDER,
                         'border_width': 1, 'border_radius': 6},
    'Rectangle::card': {'background_color': CARD, 'border_color': BORDER,
                        'border_width': 1, 'border_radius': 6},
    'Rectangle::card:selected': {'border_color': color('568cce')},
    'Rectangle::inset': {'background_color': CARD, 'border_color': BORDER,
                         'border_width': 1, 'border_radius': 4},
    'Button': {'background_color': color('333e4e'), 'border_color': color('49596e'),
               'border_width': 1, 'border_radius': 4, 'padding': 5},
    'Button:hovered': {'background_color': color('405169'), 'border_color': color('7399c5')},
    'Button:pressed': {'background_color': color('223449')},
    'Button:disabled': {'background_color': color('232c37'), 'border_color': BORDER},
    'Button.Label': {'color': TEXT, 'font_size': 14},
    'Button.Label:disabled': {'color': color('65758b')},
    'Button::primary': {'background_color': BLUE, 'border_color': color('5681cc')},
    'Button::primary:hovered': {'background_color': color('316bc1')},
    'Button::primary:pressed': {'background_color': color('1e4080')},
    'Button::primary:disabled': {'background_color': color('253a57'), 'border_color': BORDER},
    'Button::move': {'background_color': color('2d4667'), 'border_color': color('476689')},
    'Button::object_name': {'background_color': 0, 'border_width': 0, 'padding': 0},
    'Button::object_name:hovered': {'background_color': color('283b51')},
    'Button::icon': {'background_color': 0, 'border_width': 0, 'padding': 0},
    'Button::icon:hovered': {'background_color': color('304054')},
    'Field': {'background_color': CARD, 'border_color': color('425061'),
              'border_width': 1, 'border_radius': 4, 'padding': 5,
              'color': TEXT, 'font_size': 13},
    'Field:disabled': {'color': DIM},
    'ComboBox': {'background_color': CARD, 'border_color': color('425061'),
                 'border_width': 1, 'border_radius': 4, 'padding': 5},
    'ComboBox.Label': {'color': TEXT, 'font_size': 13},
    'ComboBox.Popup': {'background_color': PANEL, 'border_color': BORDER, 'border_width': 1},
    'ComboBox.Item:hovered': {'background_color': color('2d4667')},
    'ScrollingFrame': {'background_color': 0, 'border_width': 0,
                       'secondary_color': color('465465'), 'scrollbar_size': 7,
                       'scrollbar_color': color('465465'), 'scrollbar_border_radius': 4},
    'Separator': {'color': BORDER, 'border_width': 1},
    'Tooltip': {'background_color': PANEL, 'color': TEXT, 'font_size': 13, 'padding': 8},
}


def badge_style(state):
    fg, bg = {'VISIBLE': (GREEN, '1c4332'), 'LOST': (RED, '442d2b'),
              'HELD': (color('a9cfff'), '273c58')}[state]
    return {'background_color': color(bg), 'border_color': fg,
            'border_width': 1, 'border_radius': 4}, {'color': fg, 'font_size': 11}

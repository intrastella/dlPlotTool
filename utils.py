import logging
from enum import Enum
from pathlib import Path
from typing import Union

import toml
from box import Box

import matplotlib as mpl
import numpy as np
import seaborn as sns

import plotly.graph_objs as go


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class PlotType(ExtendedEnum):
    ParamsFig = "ParamsFig"
    LossFig = "LossFig"


class Schema(ExtendedEnum):
    DarkBlue = "darkblue"
    LightBlue = "lightblue"
    LightGreen = "lightgreen"
    DarkRed = "darkred"
    LightOrange = "lightorange"
    Violet = "violet"


class memorize(dict):
    def __init__(self, func):
        super().__init__()

        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, *key):
        result = self[key] = self.func(*key)
        return result


def get_config():
    """
    returns a toml dict as instance from box, i.e. dot notation
    """
    cwd = Path(__file__).resolve().parent
    location = cwd / 'fig_conf.toml'
    with open(location) as f:
        config = Box(toml.load(f))
    return config


def parser(size: int):
    # 1. find section : user data
    # 2. make everything after that empty
    # 3. add new css lines

    cwd = Path().absolute()

    with open(cwd/'assets/style.css') as f_input:
        css = f_input.read()

    template = '\n\n.lossWindow{\n    position: relative;\n    top: 10%;\n    margin: 2%;\n    ' \
             'letter-spacing: 5px;\n    width: replace%;\n    background-color: #141619;\n    display: inline-block;\n}\n'

    fig = '\n\n.loss-fig{\n    position: relative;\n    background-color: #141619;\n    padding: 0% 0% 0% 2%;\n    ' \
          'margin: 0% 5% 5% 5%;\n    letter-spacing: .5px;\n    max-width: 100%;\n    max-height: 100%;\n    ' \
          'height: auto;\n    width: auto\\9;\n}\n'

    a, value, c = template.partition('replace')
    window = a + f'{size}' + c

    head, sep, _ = css.partition('\n/* USER */\n\n')
    new_css = head + sep + window + fig

    with open(cwd/'assets/style.css', 'w') as f_output:
        f_output.write(new_css)


def get_interpolation_data(x, y, deg, inter_step):
    x_steps = x[::inter_step]
    data_len = len(x_steps)
    y_avg = [np.sum(y[i*inter_step: (i + 1)*inter_step])/inter_step for i in range(data_len)]

    z = np.polyfit(x_steps, y_avg, deg)
    f = np.poly1d(z)
    x_new = np.linspace(1, np.max(x), 50)
    y_new = f(x_new)
    return x_new, y_new


def hex_to_rgb(values):
    rgb = []
    for value in values:
        value = value.lstrip('#')
        lv = len(value)
        rgb.append(list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))
    return rgb


class ColorPicker:

    def __init__(self, color_schema):
        self.color_schema = color_schema
        self.rgb = None

    def color_range(self, n):
        n += 2

        if self.color_schema == Schema.DarkBlue.value:
            cmap = mpl.colormaps['twilight']
            indices = np.linspace(30, 215, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]

        if self.color_schema == Schema.LightBlue.value:
            map = sns.color_palette("blend:#7AB,#EDA", n).as_hex()
            # Mark or Crest
            self.rgb = hex_to_rgb(map)

        if self.color_schema == Schema.LightGreen.value:
            cmap = mpl.colormaps['viridis']
            indices = np.linspace(60, 255, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]

        if self.color_schema == Schema.LightOrange.value:
            cmap = mpl.colormaps['plasma']
            indices = np.linspace(100, 255, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]

        if self.color_schema == Schema.Violet.value:
            cmap = mpl.colormaps['viridis']
            indices = np.linspace(0, 70, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]


def col_tester(n):
    widths = np.array([0.5] * n)

    for c in ['darkblue', 'lightblue', 'lightgreen', 'lightorange']:

        picker = ColorPicker(c)
        picker.color_range(n)
        schema = picker.rgb
        color = [('rgb({r}, {g}, {b})').format(r=c[0], g=c[1], b=c[2]) for c in schema]

        y = np.array([10] * n)
        data = dict(x=np.cumsum(widths)-widths, width=widths, y=y, marker=dict(color=color))

        fig = go.Figure(go.Bar(**data))
        fig.update_xaxes(
            tickvals=np.cumsum(widths) - widths / 2,
            ticktext=np.arange(0, n+1, 1))
        fig.update_layout(xaxis_title=c)
        fig.show()


if __name__ == '__main__':
    col_tester(10)

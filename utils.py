from enum import Enum
from functools import partial
from pathlib import Path

import toml
import torch
from box import Box

import matplotlib as mpl
import numpy as np
import seaborn as sns


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class PlotType(ExtendedEnum):
    ParamsFig = "ParamsFig"
    LossFig = "LossFig"
    WeightFig = "WeightFig"


class Schema(ExtendedEnum):
    LightBlue = "lightblue"
    LightGreen = "lightgreen"
    LightOrange = "lightorange"
    Violet = "violet"


class memorize(dict):
    def __init__(self, func):
        super().__init__()

        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

    def __get__(self, instance, owner):
        return partial(self, instance)


def get_config():
    """
    returns a toml dict as instance from box, i.e. dot notation
    """
    cwd = Path(__file__).resolve().parent
    location = cwd / 'dlFigures/fig_conf.toml'
    with open(location) as f:
        config = Box(toml.load(f))
    return config


def parser(size: int):
    # 1. find section : user data
    # 2. make everything after that empty
    # 3. add new css lines

    cwd = Path(__file__).resolve().parent

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

    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if isinstance(x, list):
        x = np.array(x)

    if isinstance(y, torch.Tensor):
        y = y.numpy()

    if isinstance(y, list):
        y = np.array(y)

    x_steps = x[::inter_step]
    data_len = len(x_steps)
    y_avg = [np.sum(y[i*inter_step: (i + 1)*inter_step])/inter_step for i in range(data_len)]

    z = np.polyfit(x_steps, y_avg, deg)
    f = np.poly1d(z)
    x_new = np.linspace(1, np.max(x), 50)
    y_new = f(x_new)
    return x_new, y_new


class ColorPicker:

    def __init__(self, color_schema):
        self.color_schema = color_schema
        self.rgb = None

    def color_range(self, n):
        n += 2
        if self.color_schema == Schema.LightBlue.value:
            # Mark or Crest
            self.rgb = sns.color_palette("blend:#7AB,#EDA", n)

        if self.color_schema == Schema.LightGreen.value:
            cmap = mpl.colormaps['viridis']
            indices = np.linspace(60, 255, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]

        if self.color_schema == Schema.LightOrange.value:
            self.rgb = sns.color_palette("YlOrBr")

        if self.color_schema == Schema.Violet.value:
            cmap = mpl.colormaps['viridis']
            indices = np.linspace(0, 70, n)
            self.rgb = [cmap.colors[int(i)] for i in indices][1: n - 1]

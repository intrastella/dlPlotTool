import copy
import time
import unittest
from pathlib import Path

import numpy as np
import plotly
import toml
import torch
from box import Box
from torch.distributions import VonMises

cwd = Path(__file__).resolve().parent
file_func = cwd.parent

target = __import__("plotlyFigs")


def TestWightPLot(size):
    if size == 'large':
        weights_path = cwd / 'data/test_data/weights.pt'
    else:
        weights_path = cwd / 'data/test_data/test_weights.pt'

    weight = torch.load(weights_path, map_location=torch.device('cpu'))

    WeightFig = target.WeightFig
    weightplot = WeightFig(weight)

    exp = "exp0001"
    layer = "(0)Conv2d"
    step = 5
    idx = 5

    # dist, xaxis_range = weightplot._get_weight_dist(exp, layer, step, idx)
    weightplot.get_fig()
    plotly.offline.iplot(weightplot.fig)

    # self.assertEqual(torch.min(dist), 0) assert torch.min(dist) >= 0


def TestParamsPLot():
    location = cwd / 'data/test_data/figure_data.toml'
    with open(location) as f:
        config = Box(toml.load(f))

    data = config.ParamsPlot

    ParamsFig = target.ParamsFig
    paras = ParamsFig(data)
    paras.get_fig()
    plotly.offline.iplot(paras.fig)


def TestLossPLot():
    location = cwd / 'data/test_data/figure_data.toml'
    with open(location) as f:
        config = Box(toml.load(f))

    data = config.LossPlot

    LossFig = target.LossFig
    loss = LossFig(data)
    loss.get_fig()
    plotly.offline.iplot(loss.fig)


'''class TestFigs(unittest.TestCase):
    def WP_get_weight_dist(self):
        weights_path = cwd / 'data/test_data/weights.pt'
        
        exp = "exp0001"
        layer = "(0)conv2d"
        
        target = __import__("plotlyFigs")
        WeightFig = target.WeightFig

        weight = torch.load(weights_path, map_location=torch.device('cpu'))
        weightplot = WeightFig(weight)

        result = weightplot._get_weight_dist(exp, layer)
        self.assertEqual(result, )
        
        _protected_keys = [
        "to_dict",
        "merge_update",
        ] + [attr for attr in dir({}) if not attr.startswith("_")]
        
        '''


if __name__ == '__main__':
    # unittest.main()
    # TestParamsPLot()
    TestWightPLot('large')

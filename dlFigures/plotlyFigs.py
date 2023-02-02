import copy
import logging
from functools import cached_property
from typing import Union, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
from box import Box

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import get_config, get_interpolation_data, ColorPicker, memorize


cwd = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(module)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class WindowFig:
    def __init__(self, fig_data: dict):
        self.data = fig_data
        self.config = get_config()

        self.fig = None
        self.plot = None
        self.window = None
        self.features = None
        self.feat_pos = None
        self.display_mode = None
        self._window_data = None

    def get_fig(self):
        """
        Creates a figure for your dashboard. Saved as an attribute.
        """
        raise NotImplementedError('Needs to be created by subclass.')

    def setup(self, *args):
        """
        To update your figure based on Dash.callback input.
        :param input_parameter: callback input
        :return: figure
        """
        raise NotImplementedError('Needs to be created by subclass.')

    @property
    def window_data(self):
        self._window_data = Box(dict(window=self.window,
                                     plot=self.plot,
                                     features=self.features,
                                     display_mode=self.display_mode,
                                     feat_pos=self.feat_pos,
                                     figure=self.fig))
        return self._window_data


class LossFig(WindowFig):

    '''
    interpolation needs to be fixed regarding their bounds / updated
    '''

    def __init__(self, loss_data: dict):
        super(LossFig, self).__init__(loss_data)

        self.display_mode = self.config.LossFig.display_mode
        self.features = self.config.LossFig.features
        self.window = self.config.LossFig.window
        self.plot = self.config.LossFig.name
        self.fig = make_subplots(specs=[[{"secondary_y": False}]])

    def _update_figure(self, xlimit, ylimit):
        self.fig.data[2].update(xaxis='x2')
        self.fig.data[3].update(xaxis='x2')

        self.fig.update_yaxes(tickvals=np.linspace(0, np.round(ylimit), 5), zeroline=True, **self.config.LossFig.yaxis)
        self.fig.update_xaxes(tickvals=np.arange(1, np.round(xlimit), .5), zeroline=True, **self.config.LossFig.xaxis)

        self.fig.update_layout(**self.config.LossFig.data)

    def get_fig(self):
        model_ids = list(self.data.keys())
        self.setup(model_ids[0])

    @memorize
    def setup(self, exp):
        if exp:
            self.fig = None
            self.fig = make_subplots(specs=[[{"secondary_y": False}]])

            xlimit = 0
            ylimit = 0

            for mode in ['train', 'validation']:

                x = self.data[exp][mode].step
                y = self.data[exp][mode].loss

                xlimit = max(xlimit, np.max(x))
                ylimit = max(ylimit, np.max(y))

                true_loss = go.Scatter(
                    x=x,
                    y=y,
                    **self.config.LossFig.traceVals[mode].dataplot)

                x_new, y_new = get_interpolation_data(x, y, **self.config.LossFig.traceVals[mode].interpolation)
                inter_loss = go.Scatter(
                    x=x_new,
                    y=y_new,
                    **self.config.LossFig.traceVals[mode].interplot)

                self.fig.add_trace(true_loss, secondary_y=False)
                self.fig.add_trace(inter_loss, secondary_y=False)

            self._update_figure(xlimit, ylimit)

            logger.info(f"Figure for Loss Plot building with {exp} is done.")

        return self.fig


class ParamsFig(WindowFig):

    def __init__(self, para_data: dict):
        super(ParamsFig, self).__init__(para_data)

        self.display_mode = self.config.ParamsFig.display_mode
        self.window = self.config.ParamsFig.window
        self.plot = self.config.ParamsFig.name
        self.fig = go.Figure()

        self.color_scheme = self.config.ParamsFig.color_scheme

        self.frame_data = None
        self.changed_data = None
        self.max_val = None
        self.min_val = None
        self.fig_color = None

    def _set_graph(self, graph: int, y_idx: int):

        self.fig.add_trace(go.Scatter(
            x=self._parse_ticks(),
            y=self.changed_data.iloc[graph, :],
            customdata=self.frame_data.iloc[graph, :],
            name=self.changed_data.index[graph],
            yaxis=f"y{y_idx}",
            marker=dict(size=15, color=self.fig_color[graph]+',1)'),
            line=dict(color=self.fig_color[graph]+',0.4)'),
            hovertemplate="<br>".join([
                "%{x}: %{customdata}",
            ])
        ))

    def _adjusted_vals(self):
        def normalizer(x, key): return (x - self.min_val[key]) / (self.max_val[key] - self.min_val[key])
        def new_range(x, key): return normalizer(x, key) * (self.max_val['loss'] - self.min_val.iloc[-1]) + self.min_val['loss']

        for key in self.max_val.keys():
            if key != 'loss':
                self.changed_data[[key]] = self.frame_data[[key]].applymap(new_range, key=key)

    def _set_labels(self, ticks):
        return ['%.e' % x if x < 1 else np.round(x, 2) for x in ticks]

    def _set_legend(self):
        labels_to_show_in_legend = self.frame_data.index.to_list()
        for trace in self.fig['data']:
            if not trace['name'] in labels_to_show_in_legend:
                trace['showlegend'] = False

    def _format_color(self, n_graphs):
        picker = ColorPicker(self.color_scheme)
        picker.color_range(n_graphs)
        schema = picker.rgb
        self.fig_color = [('rgba({r}, {g}, {b}').format(r=c[0], g=c[1], b=c[2]) for c in schema]

    def _set_para_axis(self, key: str, idx: int, position: float):
        if key == 'loss':
            new_axis = {f'yaxis{idx}': dict(
                tickvals=np.linspace(self.min_val[key], self.max_val[key] + 1, 5),
                position=position,
                showgrid=False,
                showticklabels=False,
                **self.config.ParamsFig.new_axis
            )}
            self.fig.update_layout(**new_axis)

        self.fig.add_trace(go.Scatter(
            x=[self._parse_ticks(key)] * len(list(self.min_val.keys())),
            y=np.linspace(0, 5, 5),
            text=self._set_labels(np.linspace(self.min_val[key], self.max_val[key], 5)),
            yaxis=f"y",
            **self.config.ParamsFig.anno
        ))

    def _prep_data(self):
        df = pd.DataFrame(self.data)
        df = df.T
        columns = df.columns.to_numpy()
        idx = np.where(columns == 'loss')
        columns = np.delete(columns, idx)
        columns = np.append(columns, ['loss'])
        df = df[columns]
        self.frame_data = df
        self.max_val = df.max(axis=0)
        self.min_val = df.min(axis=0)

        # values must fit axes
        for key in self.max_val.keys():
            if self.max_val[key] - self.min_val[key] == 0:
                self.max_val[key] = self.max_val[key] * (1 + 2)
                self.min_val[key] = np.max(self.min_val[key] * (1 - 2), 0)

        self.changed_data = copy.deepcopy(self.frame_data.round(2))

    def _parse_ticks(self, key: str = None):
        if key:
            new_key = []
            for w in key.split('_'):
                new_key.append(w[0].upper() + w[1:])
            return (' ').join(new_key)

        else:
            parsed = []
            for l in self.frame_data.columns.to_list():
                name = []
                for w in l.split('_'):
                    name.append(w[0].upper() + w[1:])
                parsed.append((' ').join(name))
            return parsed

    def _prep_layout(self):
        (n_graphs, n_params) = self.frame_data.shape
        self.fig.update_layout(
            xaxis=dict(
                showgrid=True,
                showticklabels=True,
                ticktext=self._parse_ticks(),
                **self.config.ParamsFig.layout.xaxis
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                tickvals=np.linspace(0, 5, 5)
            ))

        self.fig.update_yaxes(visible=False)

        self.fig.update_layout(
            **self.config.ParamsFig.data
        )

        self._format_color(n_graphs)

        for idx, axis in enumerate(list(self.max_val.keys())):
            self._set_para_axis(axis, idx + 1, idx / (n_params - 1))

    def get_fig(self):
        self._prep_data()
        self._adjusted_vals()
        self._prep_layout()

        for graph in range(self.frame_data.shape[0]):
            self._set_graph(graph, self.frame_data.shape[1])

        self._set_legend()


class WeightFig(WindowFig):

    def __init__(self, weight_data: dict):
        super(WeightFig, self).__init__(weight_data)

        self.display_mode = self.config.WeightFig.display_mode
        self.features = self.config.WeightFig.features
        self.feat_pos = self.config.WeightFig.feat_pos
        self.window = self.config.WeightFig.window
        self.plot = self.config.WeightFig.name

    def get_fig(self):
        self.setup()

    @memorize
    def setup(self, exp=None, layer=None):
        if not exp:
            exp = list(self.data.keys())[0]

        if not layer:
            layer = list(self.data[exp].weights.keys())[0]

        self.fig = None
        traces = list()

        sub = 1
        if len(self.data[exp].weights[layer]) > 150:
            sub = len(self.data[exp].weights[layer]) // 150
        if sub == 0:
            sub = 1

        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size

        steps = self.data[exp].step[:sub * 150:sub]
        resized_step = len(steps) / 150
        max_dist = list()
        min_dist = list()

        for idx, step in enumerate(steps):
            dist, _, _ = self._get_weight_dist(exp, layer, step, idx)
            dist_convolved = np.convolve(dist, kernel, mode='valid')
            max_dist.append(np.max(dist_convolved).item())
            min_dist.append(np.min(dist_convolved).item())

        for idx, step in enumerate(steps):
            dist, xaxis_range, step_range = self._get_weight_dist(exp, layer, step, idx)
            dist_convolved = np.convolve(dist, kernel, mode='valid') # only when ocillation is high, function to check that

            trace = go.Scatter3d(
                x=xaxis_range, y=step_range, z=dist_convolved,
                mode="lines",
                line=dict(colorscale='rdylbu', width=4 - 3*resized_step, color=dist_convolved, cmax=max(max_dist), cmin=min(min_dist)),
                hovertemplate="<br>".join([
                    "weight: %{x}",
                    "epoch: %{y}",
                    "p: %{z}",
                ])
            )
            traces.append(trace)

        self.fig = go.Figure(data=traces)

        self.fig.update_layout(scene=dict(
            xaxis_title='Parameter values',
            yaxis_title='Epoch',
            zaxis_title='Distribution'),
            paper_bgcolor="#2E3337",
            font_color="#99A8B2",
            margin=dict(r=20, b=10, l=10, t=10),
            title=f"Experiment: {exp.upper()} \t Layer: {layer}",
            title_y=0.95,

        )

        self.fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="#2E3337",
                gridcolor="#99A8B2",
                showbackground=True,
                zerolinecolor="#99A8B2"
            ),
            yaxis=dict(
                backgroundcolor="#2E3337",
                gridcolor="#99A8B2",
                showbackground=True,
                zerolinecolor="#99A8B2"
            ),
            zaxis=dict(
                backgroundcolor="#2E3337",
                gridcolor="#99A8B2",
                showbackground=True,
                zerolinecolor="#99A8B2", ),
        ))

        self._set_legend()

        logger.info(f"Figure for Weight Plot building with {exp} and {layer} is done.")

        return self.fig

    def _get_weight_dist(self, exp: str, layer: str, step: float, idx: int):
        sub = 1
        if len(self.data[exp].weights[layer]) > 150:
            sub = len(self.data[exp].weights[layer]) // 150
        if sub == 0:
            sub = 1

        params = self.data[exp].weights[layer][:sub * 150:sub]

        params = torch.stack(params)
        max_val = torch.max(params).item()
        min_val = torch.min(params).item()

        length = 50

        translation = abs(max_val - min_val)
        translation_factor = length / translation
        transform = lambda x: (x - min_val) * translation_factor

        translated_weight = transform(params[idx].flatten())
        new_weights = torch.round(translated_weight, decimals=0).int()
        frequencies = torch.bincount(new_weights)

        if length + 1 > len(frequencies):
            added_zeros = torch.zeros(length + 1 - len(frequencies))
            frequencies = torch.concatenate((frequencies, added_zeros))

        dist = torch.round(frequencies / torch.sum(frequencies), decimals=2)
        xaxis_range = torch.linspace(min_val, max_val, length)
        step_range = torch.ones(length) * step

        return dist, xaxis_range, step_range

    def _set_legend(self):
        for trace in self.fig['data']:
            trace['showlegend'] = False


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Union

import copy
import dash
import toml
import torch

import numpy as np

from box import Box
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, ClientsideFunction
import dash_bootstrap_components as dbc

from html_methods import _multi_select, _cool_dropdown
from dlFigures import ParamsFig, LossFig, WeightFig
from utils import PlotType, get_config, parser


__author__ = "Stella Muamba Ngufulu"
__contact__ = "stellamuambangufulu@gmail.com"
__copyright__ = "Copyright 2023, DL Plot Tools"
__date__ = "2023/01/30"
__deprecated__ = False
__license__ = "Mozilla Public License 2.0"
__maintainer__ = "developer"
__status__ = "Dev"
__version__ = "0.0.1"


cwd = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(module)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class DLPlotter:

    def __init__(self):

        self._loss = dict()
        self._parameters = dict()
        self._gradients = dict()
        self._ROC_accuracy = dict()
        self._confusion_matrix = dict()

        self.custom_plot = None
        self.config = get_config('dash_conf')

    def __str__(self):
        return f'DLPlotter( \n\n' +\
               f'\tloss: \t{self._loss} \n\n' +\
               f'\tparameters: \t{self._parameters} \n\n' +\
               f'\tgradients: \t{self._gradients} \n\n' +\
               f'\tROC accuracy: \t{self._ROC_accuracy}' + \
               f')'

    def collect_loss(self,
                     exp_id: str,
                     n_batches: int,
                     epoch: int,
                     step: int,
                     loss: float,
                     mode='train'):

        """
        Collect the loss at each step to plot the training (and/or) validation loss
        of one or several experiments with different configurations of hyperparameters.

        EXAMPLE:

        plotter = DLPlotter()
        model = MyModel()
        ...
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                output = model(x)
                loss = loss_func(output, y)

                plotter.collect_loss("exp001", len(loaders), epoch, step, loss.item(), "train")
                ...
        plotter.construct()

        :param (str) exp_id: Unique identifier of an experiment. (str)
        :param (int) n_batches: Number of batchtes = total number of samples per epoch / batch size.
        :param (int) epoch: Current number of epoch starting at 0.
        :param (int) step: Current step within an epoch starting at 0.
        :param (float) loss: Amount of loss calculated by your loss function.
        :param (str) mode: Loss of a validation run or of a training run.
        """

        if not isinstance(step, int):
            raise ValueError(f"Step must be of type int not {type(step)}.")

        if not isinstance(epoch, int):
            raise ValueError(f"Epoch must be of type int not {type(epoch)}.")

        if not isinstance(n_batches, int):
            raise ValueError(f"N Batches must be of type int not {type(n_batches)}.")

        if not isinstance(exp_id, str):
            raise ValueError(f"Experiment id must be of type str not {type(exp_id)}.")

        if not isinstance(loss, float):
            raise ValueError(f"Loss must be of type str not {type(loss)}.")

        distance = 1. / n_batches
        epoch_step = epoch + step * distance
        loss = np.round(loss, 2)

        if exp_id in self._loss:
            if mode in self._loss[exp_id]:
                self._loss[exp_id][mode].step.append(epoch_step)
                self._loss[exp_id][mode].loss.append(loss)
            else:
                self._loss[exp_id][mode] = dict(step=[epoch_step], loss=[loss])
        else:
            self._loss[exp_id] = {mode: dict(step=[epoch_step], loss=[loss])}
            self._loss = Box(self._loss)

    def collect_parameter(self,
                      exp_id: str,
                      config: dict,
                      loss: float):

        """
        Collect the loss for (each / an) experiment to plot a Hyperparameter Plot
        to compare different experiments of a model regarding their loss to find
        the best configuration of hyperparameters.

        EXAMPLE:

        plotter = DLPlotter()
        model = MyModel()
        ...
        total_loss = 0
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                output = model(x)
                loss = loss_func(output, y)
                total_loss += loss.item()
                ...
        config = dict(lr=0.001, batch_size=64, ...)
        plotter.collect_parameter("exp001"", config, total_loss / (5 * len(loader))
        plotter.construct()

        :param (str) exp_id: Unique identifier for an experiment.
        :param (dict) config: Configuration file of an experiment.
        :param (float) loss: Amount of loss calculated by your loss function.
        """

        if not isinstance(config, dict):
            raise ValueError(f"Config file must be of type int not {type(config)}.")

        if not isinstance(exp_id, str):
            raise ValueError(f"Experiment id must be of type str not {type(exp_id)}.")

        if not isinstance(loss, float):
            raise ValueError(f"Loss must be of type str not {type(loss)}.")

        if exp_id in self._parameters:
            raise ValueError(f"Model id {exp_id} already collected.")

        config['loss'] = np.round(loss, 3)

        self._parameters[exp_id] = config

    def collect_weights(self,
                        exp_id: str,
                        n_batches: int,
                        epoch: int,
                        step: int,
                        weights: dict):

        """
        Collect the weights at each step to plot the learning progress of your model
        of one or several experiments.

        EXAMPLE:

        plotter = DLPlotter()
        model = MyModel()
        ...
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                weights = dict(layer1=model.layer1.weight.detach().clone(),
                               layer2=model.layer2.weight.detach().clone(), ...)

                plotter.collect_weights("exp001", len(loader), epoch, step, weights)
                ...
        plotter.construct()

        :param (str) exp_id: Unique identifier of an experiment. (str)
        :param (int) n_batches: Number of batchtes = total number of samples per epoch / batch size.
        :param (int) epoch: Current number of epoch starting at 0.
        :param (int) step: Current step within an epoch starting at 0.
        :param (dict) weights: Weights of your model at each layer.
        """

        if not isinstance(weights, dict):
            raise ValueError(f"Weights must be of type dict not {type(weights)}.")

        if not isinstance(step, int):
            raise ValueError(f"Step must be of type int not {type(step)}.")

        if not isinstance(epoch, int):
            raise ValueError(f"Epoch must be of type int not {type(epoch)}.")

        if not isinstance(n_batches, int):
            raise ValueError(f"N Batches must be of type int not {type(n_batches)}.")

        if not isinstance(exp_id, str):
            raise ValueError(f"Experiment id must be of type str not {type(exp_id)}.")

        for key in list(weights.keys()):
            if not isinstance(weights[key], torch.Tensor):
                raise ValueError(f"Weight parameters must be of type torch tensor not {type(weights[key])}.")

        distance = 1. / n_batches
        epoch_step = epoch + step * distance

        if exp_id in self._gradients:
            self._gradients[exp_id].step.append(epoch_step)
            for key in list(weights.keys()):
                self._gradients[exp_id].weights[key].append(weights[key])

        else:
            for key in list(weights.keys()):
                weights[key] = [weights[key]]
            self._gradients[exp_id] = dict(step=[epoch_step], weights=weights)
            self._gradients = Box(self._gradients)

    def load_from_model(self, path: Union[str, Path]):
        checkpoint = torch.load(path)

    def add_plot(self, figures: list):
        self.custom_plot = figures

    def _build_progress_section(self):
        _learning_progress = list()
        if self._parameters:
            paras = ParamsFig(self._parameters)
            paras.get_fig()
            _learning_progress.append(paras)

        if self._loss:
            data_dir = cwd / "dlFigures/fig_conf.toml"
            with open(data_dir) as f:
                data = f.read()
            d = toml.loads(data)
            new_data = copy.deepcopy(d)
            new_data['LossFig']['CDropdown']['options'] = list(self._loss.keys())
            with open(data_dir, 'w') as f:
                f.write(toml.dumps(new_data))

            loss = LossFig(self._loss)
            loss.get_fig()
            _learning_progress.append(loss)

        if self._gradients:
            # update exp_ids and layers

            weights = WeightFig(self._gradients)
            weights.get_fig()
            _learning_progress.append(weights)

        if self.custom_plot:
            _learning_progress += self.custom_plot

        logger.info("Fetching figure objects for Learning Progress section completed.")

        return _learning_progress

    def _build_accuracy_section(self):
        _accuracy = list()
        return _accuracy

    def _build_output_section(self):
        _output = list()
        return _output

    def construct(self, port: int):
        dash_page = DashStruct()

        dash_page.create_section(self._build_progress_section(), **self.config.progress_section)
        dash_page.create_section(self._build_accuracy_section(), **self.config.accuracy_section)
        dash_page.create_section(self._build_output_section(), **self.config.output_section)

        dash_page.app.layout = html.Div(children=[dash_page.header] + dash_page.page_section)
        dash_page.show(port)


class DashStruct:
    header = html.Header(children=[

                html.H1(children="Deep Learning Analyser", id="in-logo"),
                html.P(children="DLA", id="logo"),
                html.P(children="Analyze the behavior of your DL models"
                                " in a comprehensive and easy to use way.",
                       className="header-description", id="dscript"),
                html.Nav(html.Div(children=[
                                       html.A(href="#learning_progress", children="Learning Progress".upper()),
                                       html.A(href="#accuracy", children="Accuracy".upper()),
                                       html.A(href="#sample_outputs", children="Model Output".upper()),
                                       ], className="topnav"), id="navcontainer")

    ], className="header", id="app-page-header")
    footer = None

    def __init__(self):
        self.config = get_config()
        self.app = Dash(__name__,
                        external_stylesheets=[dbc.themes.BOOTSTRAP],
                        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
        self.app.title = "DL Plots"

        self.app.clientside_callback(
            ClientsideFunction("clientside", "responsiveNav"),
            Output("navcontainer", "data-loaded"),
            [Input("navcontainer", "id")])

        self.app.clientside_callback(
            ClientsideFunction("clientside", "stickyHeader"),
            Output("app-page-header", "data-loaded"),
            [Input("app-page-header", "id"), Input("dscript", "id")])

        self.app.callback(
            dash.dependencies.Output(PlotType.LossFig.value, 'figure'),
            [dash.dependencies.Input('model-id', 'value')])(self._single_drop_update)

        self.app.callback(
            dash.dependencies.Output(PlotType.WeightFig.value, 'figure'),
            [dash.dependencies.Input('w-model-id1', 'value'), dash.dependencies.Input('w-model-id2', 'value')])(
            self._multi_drop_update)

        self.page_section = list()
        self._single_drop_figs = list()
        self._multi_drop_figs = list()

    def window(self,
               window: str,
               plot: str,
               features: Union[str, list] = None,
               display_mode: str = 'single',
               feat_pos: str = None,
               **kwargs) -> dbc.Col:

        dis_feat = list()

        if features:
            if features == 'multi_select':
                dis_feat.append(_multi_select(window+features, *list(self.config[plot].MSelect.values())))

            if features == 'cool_dropdown':
                dis_feat.append(_cool_dropdown(self.config[plot].CDropdown))

        return self._structure(window,
                               [html.H2(**self.config[plot].Div)],
                               [dcc.Graph(**self.config[plot].Graph, **kwargs, responsive=True)],
                               display_mode,
                               dis_feat,
                               feat_pos)

    @staticmethod
    def _structure(window: str,
                   title: list,
                   graph: list,
                   display_mode: str,
                   dis_feat: list,
                   feat_pos=None) -> dbc.Col:

        if display_mode == 'single':
            rows = [dbc.Row(dbc.Col(html.Div(children=title)))]
            if feat_pos:
                rows.append(dbc.Row([dbc.Col(html.Div(graph), width=6), dbc.Col(html.Div(dis_feat), width=6)]))
            else:
                rows += [dbc.Row(dbc.Col(dis_feat, width=12)), dbc.Row(dbc.Col(graph, width=12))]
            section = dbc.Col(dbc.Card(dbc.CardBody(rows), className=window), width=12)

        else:
            section = dbc.Col(dbc.Card(dbc.CardBody(title + dis_feat + graph), className=window), width=6)

        return section

    def _single_drop_update(self, input_parameter):
        """
        Collect data in DLPlotter and send the instantiated class to DashStruct.
        S.t.: w/ fig.setup(input_parameter) returns a new fig to update current fig
        :return: matplotlib and any subclass figures like plotly
        """
        for fig in self._single_drop_figs:
            if fig.window_data.features == "cool_dropdown":
                return fig.setup(input_parameter)

    def _multi_drop_update(self, exp_id, layer):
        """
        Collect data in DLPlotter and send the instantiated class to DashStruct.
        S.t.: w/ fig.setup(input_parameter) returns a new fig to update current fig
        :return: matplotlib and any subclass figures like plotly
        """
        for fig in self._multi_drop_figs:
            if fig.window_data.features == 'multi_select':
                return fig.setup(exp_id, layer)

    def set_prop(self, fig):
        """
        When you want to add your own figures as subclass of WindowFig this method will add css properties
        """
        if fig.window_data.features == "cool_dropdown":
            self._single_drop_figs.append(fig)

        if fig.window_data.features == "multi_select":
            self._multi_drop_figs.append(fig)

        if not fig.window_data.plot in PlotType.values():
            size = 50
            if fig.window_data.display_mode == 'single':
                size = 100
            parser(size)

    def create_section(self,
                       figs: list,
                       intro: dict,
                       section_id: str,
                       section_class: str):

        containers = []
        section = [html.P(**intro)]

        if len(figs) > 0:
            for i, fig in enumerate(figs):
                self.set_prop(fig)

                if fig.window_data.display_mode == 'single':
                    section.append(dbc.Row(children=self.window(**fig.window_data)))

                else:
                    if len(containers) < 1:
                        containers.append(self.window(**fig.window_data))

                    else:
                        containers.append(self.window(**fig.window_data))
                        section.append(dbc.Row(children=containers, className="double"))
                        containers = []

        else:
            section.append(html.P(children="No data has been collected.", className="empty_frame"))
            section.append(dbc.Row(children=containers))

        self.page_section.append(dbc.Card(dbc.CardBody(section), id=section_id, className=section_class))

    def show(self, port):
        self.app.run_server(debug=False, port=port, use_reloader=False)

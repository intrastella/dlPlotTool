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

from html_methods import _dropdown, _select, _cool_dropdown
from plotlyFigs import ParamsFig, LossFig, WeightFig
from utils import PlotType, get_config, parser


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

        self.custom_plot = None

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

        :param exp_id: Unique identifier of an experiment. Type = str.
        :param n_batches: Number of batchtes = total number of samples per epoch / batch size. Type = int.
        :param epoch: Current number of epoch. Type = int.
        :param step: Current step within an epoch. Type = int.
        :param loss: Amount of loss calculated by your loss function. Type = float.
        :param mode: Loss of a validation run or of a training run. Type = str.
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

        :param exp_id: Unique identifier for an experiment. Type = str.
        :param config: Configuration file of an experiment. Type = dict.
        :param loss: Amount of loss calculated by your loss function. Type = float.
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

    def _build_plt(self):

        figs = list()

        if self.custom_plot:
            figs += self.custom_plot

        if self._parameters:
            paras = ParamsFig(self._parameters)
            paras.get_fig()
            figs.append(paras)

        if self._loss:
            data_dir = cwd / "fig_conf.toml"
            with open(data_dir) as f:
                data = f.read()
            d = toml.loads(data)
            new_data = copy.deepcopy(d)
            new_data['LossFig']['CDropdown']['options'] = list(self._loss.keys())
            with open(data_dir, 'w') as f:
                f.write(toml.dumps(new_data))

            loss = LossFig(self._loss)
            loss.get_fig()
            figs.append(loss)

        if self._gradients:
            weights = WeightFig(self._gradients)
            weights.get_fig()
            figs.append(weights)

        logger.info("Fetching figure objects for Dashboard completed.")

        return figs

    def construct(self, port: int):
        dash_img = DashStruct(self._build_plt())
        dash_img.build()
        dash_img.show(port)


class DashStruct:
    header = html.Header(children=[
                html.H1(children="Deep Learning Analyser", className="header-title"),

                html.P(children="Analyze the behavior of your DL models"
                                " in a comprehensive and easy to use way.",
                       className="header-description", id="dscript")], className="header", id="app-page-header")
    footer = None

    def __init__(self, data):
        self.figs = data
        self.config = get_config()
        self.app = Dash(__name__,
                        external_stylesheets=[dbc.themes.BOOTSTRAP],
                        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
        self.app.title = "DL Plots"

        self.app.callback(
            dash.dependencies.Output(PlotType.LossFig.value, 'figure'),
            [dash.dependencies.Input('model-id', 'value')])(self.update_graphs)

        '''self.app.callback(
            dash.dependencies.Output(PlotType.WeightFig.value, 'figure'),
            [dash.dependencies.Input('w-model-id', 'value')])(self.update_graphs1)'''

        self.app.clientside_callback(
            ClientsideFunction("clientside", "stickyHeader"),
            Output("app-page-header", "data-loaded"),
            [Input("app-page-header", "id"), Input("dscript", "id")])

    def window(self,
               window: str,
               plot: str,
               features: Union[str, list] = None,
               display_mode: str = 'single',
               **kwargs):

        content = [html.H2(**self.config[plot].Div), dcc.Graph(**self.config[plot].Graph, **kwargs, responsive=True)]

        if features:
            if isinstance(features, str):
                features = [features]

            for feature in features:
                content_feat = list()

                if feature == 'multi_select':
                    content_feat = _dropdown(**self.config[plot].Dropdown)

                if feature == 'cool_dropdown':
                    content_feat = _cool_dropdown(**self.config[plot].CDropdown)

                if feature == 'select':
                    content_feat = _select(**self.config[plot].Select)

                if display_mode == 'single':
                    content = [dbc.Col(content), dbc.Col(content_feat)]
                else:
                    content = content[:1] + content_feat + content[1:]

        return html.Div(children=content, className=window)

    def update_graphs(self, input_parameter):
        """
        Collect data in DLPlotter and send the instantiated class to DashStruct.
        S.t.: w/ fig.setup(input_parameter) returns a new fig to update current fig
        :return: matplotlib and any subclass figures like plotly
        """
        for fig in self.figs:
            if fig.window_data.features:
                return fig.setup(input_parameter)

    def _multi_drop_update(self, input_parameter):
        """
        Collect data in DLPlotter and send the instantiated class to DashStruct.
        S.t.: w/ fig.setup(input_parameter) returns a new fig to update current fig
        :return: matplotlib and any subclass figures like plotly
        """
        fig = self.figs[2]
        if fig.window_data.features:
            return fig.setup(input_parameter)

    def set_prop(self, fig):
        """
        When you want to add your own figures as subclass of WindowFig this method will add css properties
        """
        if not fig.window_data.plot in PlotType.values():
            size = 50
            if fig.window_data.display_mode == 'single':
                size = 100
            parser(size)

    def build(self):
        count = 0
        sections = []
        page = [self.header]

        for i, fig in enumerate(self.figs):
            self.set_prop(fig)

            # needs to be improved so far left orientation

            if fig.window_data.display_mode == 'single':
                page.append(dbc.Row(children=sections))
                page.append(dbc.Row(children=[self.window(**fig.window_data)]))
                sections = []

            else:
                if count < 2:
                    sections.append(dbc.Col(self.window(**fig.window_data), width=6))
                    if i + 1 == len(self.figs):
                        page.append(dbc.Row(children=sections, className="section1"))
                    count += 1
                else:
                    page.append(dbc.Row(children=sections))
                    sections = [dbc.Col(self.window(**fig.window_data), width=6)]
                    count = 0

        self.app.layout = html.Div(children=page)

    def show(self, port):
        self.app.run_server(debug=False, port=port, use_reloader=False)
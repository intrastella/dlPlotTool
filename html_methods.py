import json
from pathlib import Path
from typing import Union, List

import copy
import dash
import toml
import torch
from box import Box
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_defer_js_import as dji

from plotlyFigs import ParamsFig, LossFig
from utils import PlotType, get_config, parser


def _multi_select(className: str, *args):

    rows = list()

    for arg in args:
        drop = dcc.Dropdown(**arg)
        rows.append(dbc.Row(dbc.Col(drop, width=12)))

    return dbc.Card(dbc.CardBody(rows), className=className)


def _cool_dropdown(kwargs):
    return dcc.Dropdown(**kwargs)

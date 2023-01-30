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


# nav , input , select, ul , il


def _dropdown(options: List[str],
              placeholder: str) -> list:

    return [dbc.DropdownMenu(
        color="#141619",
        label=placeholder,
        class_name="menu",
        toggleClassName="items",
        menu_variant="dark",
        children=html.Div([
            html.Div(dbc.DropdownMenuItem(op, class_name="item"), className="select-item") for op in options
        ], className="drop"),
    )]


def _select(options: list,
            id: str,
            className: List[str],
            placeholder: str) -> list:
    selections = []
    # [html.Ul(selections, className=className[1])]
    for val in options:
        selections.append(html.Li(children=val, className=className[1]))

    return [dcc.Input(list=selections,
                     type="text", id=id,
                     placeholder=placeholder,
                     className=className[0])]


def _cool_dropdown(options: list,
                   className: List[str],
                   placeholder: str,
                   id: str) -> list:
    return [dcc.Dropdown(
        options=options,
        placeholder=placeholder,
        className=className,
        id=id)]

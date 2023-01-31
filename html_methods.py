from dash import dcc
import dash_bootstrap_components as dbc


def _multi_select(className: str, *args):

    rows = list()

    for arg in args:
        drop = dcc.Dropdown(**arg)
        rows.append(dbc.Row(dbc.Col(drop, width=12)))

    return dbc.Card(dbc.CardBody(rows), className=className)


def _cool_dropdown(kwargs):
    return dcc.Dropdown(**kwargs)

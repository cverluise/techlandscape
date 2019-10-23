import pandas as pd
import plotly.express as px
from techlandscape.config import Config

config = Config()
color_single = config.color_single
color_sequence = config.color_sequence


def _prep_data(df, groups, f, norm_axis):
    """
    Return <df> grouped by <groups> with values aggregated by <f> (e.g. len for count,
    sum for sum, etc) and normalized along the <norm_axis>st/nd.
    :param df: pd.Dataframe
    :param groups: list
    :param f: function
    :param norm_axis: int, in [0, 1]
    :return: pd.DataFrame
    """
    for e in groups:
        assert e in df.columns

    tmp = df.groupby(groups).apply(f)
    if f == sum:
        tmp = tmp.drop(tmp.index.names, axis=1).squeeze()

    tmp = tmp.rename("count").reset_index()
    tmp["normalized count"] = tmp.groupby(groups[norm_axis]).transform(
        lambda x: x / sum(x) if norm_axis == 0 else x / max(x)
    )["count"]

    return tmp


def fig_wrapper(df, mode, groups, f, normalize=False, norm_axis=0, **kwargs):
    """
    TODO
    :param df: pd.DataFrame
    :param mode: str, in ["bar", "line"] (nb: could be extended to other px modes)
    :param groups: list,
    :param f: function, in [sum, len]
    :param normalize: bool,
    :param norm_axis: int, in [0,1]
    :param kwargs: key worded arguments that will be passed to the fig object
    :return: px.fig, pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(groups, list)
    assert mode in ["bar", "line"]

    tmp = _prep_data(df, groups, f, norm_axis)

    y = "normalized count" if normalize else "count"
    if len(groups) == 2:
        kwargs.update({"color": groups[1]})
    if len(groups) == 1:
        tmp = tmp.sort_values(y)  # aesthetic purpose only

    fig = px.bar if mode == "bar" else px.line
    fig = fig(
        tmp,
        x=groups[0],
        y=y,
        color_discrete_sequence=color_sequence
        if len(groups) == 2
        else color_single,
        **kwargs
    )

    fig.update_layout(legend={"orientation": "h", "y": -0.15})
    return fig

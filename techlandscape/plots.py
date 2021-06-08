import pandas as pd
import warnings
import plotly.express as px
from techlandscape.lib import COLOR_UNIQUE, COLOR_SEQUENCE
from typing import List


def fig_wrapper(
    df: pd.DataFrame,
    mode: str,
    groups: List,
    func: callable,
    normalize: bool = False,
    norm_axis: int = 0,
    **kwargs
):
    """
    Return a formatted plotly fig

    Values:
        - `mode`: "bar", "line"
        - `norm_axis`: 0, 1
    """

    def _prep_data(df: pd.DataFrame, groups: List, func: callable, norm_axis: int):
        """
        Return `df` grouped by `groups` with values aggregated by `func` (e.g. len for count,
        sum for sum, etc) and normalized along the `norm_axis` st/nd.

        Values:
            - `norm_axis`: 0, 1
        """
        for e in groups:
            assert e in df.columns

        tmp = df.groupby(groups).apply(func)
        if func == sum:
            tmp = tmp.drop(tmp.index.names, axis=1).squeeze()

        tmp = tmp.rename("count").reset_index()
        tmp["normalized count"] = tmp.groupby(groups[norm_axis]).transform(
            lambda x: x / sum(x) if norm_axis == 0 else x / max(x)
        )["count"]

        return tmp

    warnings.warn(
        "Deprecated. Will be removed in the near future. Use SciPlots instead.",
        DeprecationWarning,
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(groups, list)
    assert mode in ["bar", "line"]

    tmp = _prep_data(df, groups, func, norm_axis)

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
        color_discrete_sequence=COLOR_SEQUENCE if len(groups) == 2 else COLOR_UNIQUE,
        **kwargs
    )

    fig.update_layout(legend={"orientation": "h", "y": -0.15})
    return fig

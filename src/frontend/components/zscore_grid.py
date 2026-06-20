"""Wrapped z-score tile grid for all metrics (overview beside radar)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from frontend.components.radar_chart import RADAR_FIGURE_HEIGHT

_DIVERGING_SCALE: list[list[float | str]] = [
    [0, "rgb(31, 119, 180)"],
    [0.5, "rgb(255, 255, 255)"],
    [1, "rgb(255, 140, 0)"],
]


def _abbrev(name: str, max_len: int = 10) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def make_zscore_grid_figure(
    feature_cols: list[str],
    z_values: np.ndarray,
    batch_values: np.ndarray,
    medians: np.ndarray | pd.Series,
    *,
    template: str = "plotly_white",
    title: str = "All metrics (z vs population)",
    height: int | None = None,
) -> go.Figure:
    """
    Square-ish wrapped heatmap: one tile per feature (alphabetical ``feature_cols``).

    ``z_values`` should match the radar scale (typically clipped to [-3, 3]).

    ``height`` defaults to ``RADAR_FIGURE_HEIGHT`` so the grid aligns with the radar chart.
    """
    fig_height = RADAR_FIGURE_HEIGHT if height is None else height
    n = len(feature_cols)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template=template, height=fig_height)
        return fig

    z_flat = np.asarray(z_values, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(batch_values, dtype=np.float64).reshape(-1)
    if isinstance(medians, pd.Series):
        med_arr = np.array([float(medians[f]) for f in feature_cols], dtype=np.float64)
    else:
        med_arr = np.asarray(medians, dtype=np.float64).reshape(-1)
    if not (len(z_flat) == len(b_flat) == len(med_arr) == n):
        msg = "feature_cols, z_values, batch_values, medians must align in length"
        raise ValueError(msg)

    ncols = max(2, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    z_grid = np.full((nrows, ncols), np.nan, dtype=np.float64)
    text: list[list[str]] = [[""] * ncols for _ in range(nrows)]
    customdata: list[list[list[float | str]]] = [
        [["", 0.0, 0.0, 0.0] for _ in range(ncols)] for _ in range(nrows)
    ]

    for i, feat in enumerate(feature_cols):
        r, c = divmod(i, ncols)
        z_grid[r, c] = z_flat[i]
        text[r][c] = _abbrev(feat)
        customdata[r][c] = [feat, float(z_flat[i]), float(b_flat[i]), float(med_arr[i])]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "z=%{customdata[1]:.3f}<br>"
        "batch=%{customdata[2]:.6g}<br>"
        "median=%{customdata[3]:.6g}"
        "<extra></extra>"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_grid,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorscale=_DIVERGING_SCALE,
            zmin=-3.0,
            zmax=3.0,
            colorbar=dict(
                title="z-score",
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
            ),
            xgap=2,
            ygap=2,
            showscale=True,
        ),
    )
    fig.update_layout(
        title=title,
        template=template,
        height=fig_height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig

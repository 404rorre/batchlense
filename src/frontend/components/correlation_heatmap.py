"""Clustered Pearson correlation heatmap for Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def correlation_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Pearson correlation of numeric feature columns."""
    return df[feature_cols].astype(float).corr(method="pearson")


def clustered_order(corr: pd.DataFrame) -> list[str]:
    """Reorder labels by hierarchical clustering on correlation distance."""
    labels = list(corr.columns)
    n = len(labels)
    if n <= 1:
        return labels
    c = corr.values.astype(float)
    np.fill_diagonal(c, 1.0)
    c = np.clip(c, -1, 1)
    dist = 1.0 - np.abs(c)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method="average")
    order = hierarchy.leaves_list(link)
    return [labels[i] for i in order]


def make_correlation_heatmap_figure(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    template: str = "plotly_white",
) -> go.Figure:
    """Heatmap with diverging colors, annotations, upper triangle masked."""
    corr = correlation_matrix(df, feature_cols)
    order = clustered_order(corr)
    corr = corr.loc[order, order]
    n = len(order)
    z = corr.values.astype(float)
    text = [[f"{v:.2f}" for v in row] for row in z]
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    z_masked = z.copy()
    z_masked[mask_upper] = np.nan
    for i in range(n):
        for j in range(n):
            if mask_upper[i, j]:
                text[i][j] = ""

    fig = go.Figure(
        data=go.Heatmap(
            z=z_masked,
            x=order,
            y=order,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=[
                [0, "rgb(31, 119, 180)"],
                [0.5, "rgb(255, 255, 255)"],
                [1, "rgb(255, 140, 0)"],
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate="%{x} vs %{y}<br>r=%{z:.3f}<extra></extra>",
        ),
    )
    fig.update_layout(
        title="Feature correlations (clustered)",
        template=template,
        xaxis={"side": "bottom", "tickangle": -45},
        yaxis={"autorange": "reversed"},
        height=max(400, 40 * n),
    )
    return fig

"""Histogram with KDE overlay and outlier markers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def make_feature_histogram_figure(
    df: pd.DataFrame,
    feature: str,
    *,
    outlier_col: str = "is_outlier",
    template: str = "plotly_white",
    n_bins: int = 30,
) -> go.Figure:
    x = df[feature].astype(float).values
    x = x[np.isfinite(x)]
    if x.size == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{feature} (no data)", template=template)
        return fig

    counts, edges = np.histogram(x, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    bin_edges = np.column_stack([edges[:-1], edges[1:]])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths * 0.95,
            name="Count",
            marker_color="rgba(100,100,120,0.6)",
            customdata=bin_edges,
            hovertemplate=(
                "low: %{customdata[0]:.6g}<br>"
                "high: %{customdata[1]:.6g}<br>"
                "count: %{y}<extra></extra>"
            ),
        ),
    )

    if x.size >= 3:
        kde = stats.gaussian_kde(x)
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        ys = kde(xs)
        ys_scaled = ys * (counts.max() / ys.max()) if ys.max() > 0 else ys
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys_scaled,
                mode="lines",
                name="Density (smooth)",
                line=dict(color="darkblue", width=2),
            ),
        )

    if outlier_col in df.columns:
        ox = df.loc[df[outlier_col] == 1, feature].astype(float)
        for val in ox:
            if np.isfinite(val):
                fig.add_vline(
                    x=val,
                    line_width=2,
                    line_dash="solid",
                    line_color="#FF8C00",
                    annotation_text="outlier",
                    annotation_position="top",
                )

    fig.update_layout(
        title=f"{feature}: distribution vs outliers",
        xaxis_title=feature,
        yaxis_title="Count",
        template=template,
        showlegend=True,
        bargap=0.05,
    )
    return fig

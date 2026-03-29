"""Radar chart comparing a batch to population mean (z-scores)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def z_scores_for_batch(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_mask: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std from full ``df``; z-scores for rows in ``batch_mask`` (mean if multiple)."""
    pop = df[feature_cols].astype(float)
    mean = pop.mean()
    std = pop.std(ddof=1).replace(0, np.nan)
    row = df.loc[batch_mask, feature_cols].astype(float).mean()
    z = (row - mean) / std
    z = z.fillna(0.0)
    z_clipped = np.clip(z.values, -3.0, 3.0)
    return mean.values, z_clipped


def make_radar_figure(
    feature_cols: list[str],
    z_reference: np.ndarray,
    z_batch: np.ndarray,
    *,
    template: str = "plotly_white",
    title: str = "Feature profile vs average batch",
) -> go.Figure:
    """``z_reference`` typically zeros (average); ``z_batch`` z-scores per feature."""
    theta = list(feature_cols) + [feature_cols[0]]
    r_ref = list(z_reference) + [z_reference[0]]
    r_bat = list(z_batch) + [z_batch[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_ref,
            theta=theta,
            fill="toself",
            name="Typical (average)",
            line_color="lightgray",
            fillcolor="rgba(200,200,200,0.3)",
        ),
    )
    fig.add_trace(
        go.Scatterpolar(
            r=r_bat,
            theta=theta,
            fill="toself",
            name="Selected batch",
            line_color="#FF8C00",
            fillcolor="rgba(255,140,0,0.35)",
        ),
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
        showlegend=True,
        title=title,
        template=template,
        height=500,
    )
    return fig

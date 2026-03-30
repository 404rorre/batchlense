"""Compare one batch to population medians."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def comparison_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_key_col: str,
    batch_value: str,
) -> pd.DataFrame:
    """Rows: features; columns: batch value, median, delta, delta_pct."""
    row = df.loc[df[batch_key_col].astype(str) == str(batch_value)]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    med = df[feature_cols].astype(float).median()
    rows = []
    for c in feature_cols:
        v = float(row[c])
        m = float(med[c])
        delta = v - m
        pct = 100.0 * delta / m if m != 0 else (np.inf if delta != 0 else 0.0)
        pct_r = round(float(pct), 2) if np.isfinite(pct) else np.nan
        rows.append(
            {
                "feature": c,
                "batch": round(v, 2),
                "median_all": round(m, 2),
                "delta": round(delta, 2),
                "delta_pct": pct_r,
            },
        )
    return pd.DataFrame(rows)


def make_delta_bar_figure(
    comp: pd.DataFrame, *, template: str = "plotly_white"
) -> go.Figure:
    """Horizontal bar chart of % vs median."""
    if comp.empty:
        return go.Figure()
    c = comp.dropna(subset=["delta_pct"])
    fig = go.Figure(
        go.Bar(
            x=c["delta_pct"],
            y=c["feature"],
            orientation="h",
            marker_color=np.where(c["delta_pct"] >= 0, "#FF8C00", "rgb(100,149,237)"),
            text=[f"{v:+.2f}%" for v in c["delta_pct"]],
            textposition="outside",
        ),
    )
    fig.update_layout(
        title="% difference vs median (all batches)",
        xaxis_title="Δ % vs median",
        template=template,
        height=max(300, 28 * len(c)),
    )
    return fig

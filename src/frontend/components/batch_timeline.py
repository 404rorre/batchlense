"""Time-ordered line + markers for one feature."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go

Resample = Literal["month", "day", "hour"]


def make_timeline_figure(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    feature: str,
    outlier_col: str = "is_outlier",
    batch_col: str | None = None,
    resample: Resample = "day",
    template: str = "plotly_white",
) -> go.Figure:
    d = df.copy()
    d["_ts"] = pd.to_datetime(d[datetime_col], errors="coerce")
    d = d.dropna(subset=["_ts"]).sort_values("_ts")
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid dates", template=template)
        return fig

    rule = {"month": "ME", "day": "D", "hour": "h"}[resample]
    g = d.set_index("_ts").groupby(pd.Grouper(freq=rule))

    xs: list[pd.Timestamp] = []
    ys: list[float] = []
    out_flags: list[bool] = []
    hover: list[str] = []

    for t, grp in g:
        if grp.empty:
            continue
        xs.append(t)
        ys.append(float(grp[feature].mean()))
        if outlier_col in grp.columns:
            out_flags.append(bool((grp[outlier_col] == 1).any()))
        else:
            out_flags.append(False)
        parts = [f"{feature} (mean): {ys[-1]:.3f}"]
        if batch_col and batch_col in grp.columns:
            parts.append("Batches: " + ", ".join(grp[batch_col].astype(str).head(5)))
        hover.append("<br>".join(parts))

    colors = ["#FF8C00" if o else "#D3D3D3" for o in out_flags]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color="rgba(80,80,100,0.8)", width=2),
            marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
    )
    fig.update_layout(
        title=f"{feature} over time ({resample})",
        xaxis_title="Time",
        yaxis_title=feature,
        template=template,
        hovermode="closest",
    )
    return fig

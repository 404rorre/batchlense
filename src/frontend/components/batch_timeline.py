"""Time-ordered line + markers for one feature."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

Resample = Literal["month", "day", "hour"]


def baseline_feature_series(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    feature: str,
    outlier_col: str = "is_outlier",
    in_control_only: bool = True,
) -> pd.Series:
    """Feature values with valid timestamps; optionally exclude flagged outliers."""
    d = df.copy()
    d["_ts"] = pd.to_datetime(d[datetime_col], errors="coerce")
    d = d.dropna(subset=["_ts"]).sort_values("_ts")
    if in_control_only and outlier_col in d.columns:
        d = d[d[outlier_col] != 1]
    return (
        d[feature]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def has_in_control_baseline(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    feature: str,
    outlier_col: str = "is_outlier",
) -> bool:
    """True when at least one non-flagged row has a finite feature value."""
    return (
        len(
            baseline_feature_series(
                df,
                datetime_col=datetime_col,
                feature=feature,
                outlier_col=outlier_col,
                in_control_only=True,
            ),
        )
        > 0
    )


def make_timeline_figure(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    feature: str,
    outlier_col: str = "is_outlier",
    batch_col: str | None = None,
    resample: Resample = "day",
    template: str = "plotly_white",
    ucl: float | None = None,
    lcl: float | None = None,
    uwl: float | None = None,
    lwl: float | None = None,
    usl: float | None = None,
    lsl: float | None = None,
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

    baseline = baseline_feature_series(
        d,
        datetime_col="_ts",
        feature=feature,
        outlier_col=outlier_col,
        in_control_only=True,
    )
    xbar_val = float(baseline.mean()) if len(baseline) > 0 else float("nan")

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
    _red = "#C62828"
    if lcl is not None and math.isfinite(lcl):
        fig.add_hline(
            y=lcl,
            line_dash="dash",
            line_color=_red,
            line_width=4,
            annotation_text=f"LCL (UEG) {lcl:.3f}",
            annotation_position="right",
        )
    if lwl is not None and math.isfinite(lwl):
        fig.add_hline(
            y=lwl,
            line_dash="dash",
            line_color=_red,
            line_width=1,
            annotation_text=f"LWL (UWG) {lwl:.3f}",
            annotation_position="right",
        )
    if math.isfinite(xbar_val):
        fig.add_hline(
            y=xbar_val,
            line_dash="solid",
            line_color="#2E7D32",
            line_width=2,
            annotation_text=f"x\u0304 {xbar_val:.3f}",
            annotation_position="right",
        )
    if uwl is not None and math.isfinite(uwl):
        fig.add_hline(
            y=uwl,
            line_dash="dash",
            line_color=_red,
            line_width=1,
            annotation_text=f"UWL (OWG) {uwl:.3f}",
            annotation_position="right",
        )
    if ucl is not None and math.isfinite(ucl):
        fig.add_hline(
            y=ucl,
            line_dash="dash",
            line_color=_red,
            line_width=4,
            annotation_text=f"UCL (OEG) {ucl:.3f}",
            annotation_position="right",
        )
    if usl is not None and math.isfinite(usl):
        fig.add_hline(
            y=usl,
            line_dash="solid",
            line_color="#1565C0",
            line_width=2,
            annotation_text=f"USL {usl:.3f}",
            annotation_position="right",
        )
    if lsl is not None and math.isfinite(lsl):
        fig.add_hline(
            y=lsl,
            line_dash="solid",
            line_color="#1565C0",
            line_width=2,
            annotation_text=f"LSL {lsl:.3f}",
            annotation_position="right",
        )
    fig.update_layout(
        title=f"{feature} over time ({resample})",
        xaxis_title="Time",
        yaxis_title=feature,
        template=template,
        hovermode="closest",
    )
    return fig


def make_production_bar_figure(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    outlier_col: str = "is_outlier",
    resample: Resample = "day",
    template: str = "plotly_white",
) -> go.Figure:
    """Stacked bar: batch counts per time bucket (normal vs flagged)."""
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
    approved: list[int] = []
    outliers: list[int] = []

    for t, grp in g:
        if grp.empty:
            continue
        xs.append(t)
        if outlier_col in grp.columns:
            o = grp[outlier_col].astype(int)
            outliers.append(int((o == 1).sum()))
            approved.append(int((o != 1).sum()))
        else:
            approved.append(len(grp))
            outliers.append(0)

    totals = [a + b for a, b in zip(approved, outliers, strict=True)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xs,
            y=approved,
            name="Normal",
            marker_color="#D3D3D3",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=xs,
            y=outliers,
            name="Flagged",
            marker_color="#FF8C00",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=totals,
            mode="text",
            text=[str(t) for t in totals],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        ),
    )
    fig.update_layout(
        title=f"Batches per period ({resample}) — stacked: normal vs flagged",
        xaxis_title="Time",
        yaxis_title="Batch count",
        barmode="stack",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

"""Production-style KPIs and optional process capability (Cp/Cpk)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def basic_kpis(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    outlier_col: str = "is_outlier",
    lof_col: str = "lof_score",
) -> dict[str, Any]:
    """Summary metrics for the dashboard."""
    n = len(df)
    if outlier_col in df.columns:
        out = int(df[outlier_col].sum())
        rate = 100.0 * out / n if n else 0.0
    else:
        out, rate = 0, 0.0

    feats = df[feature_cols].astype(float)
    per_feature: dict[str, dict[str, float]] = {}
    for c in feature_cols:
        s = feats[c]
        per_feature[c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }

    lof_min = lof_max = 0.0
    if lof_col in df.columns:
        ls = df[lof_col].astype(float)
        lof_min, lof_max = float(ls.min()), float(ls.max())

    return {
        "n_batches": n,
        "n_outliers": out,
        "outlier_rate_pct": rate,
        "per_feature": per_feature,
        "lof_min": lof_min,
        "lof_max": lof_max,
    }


def auto_spec_limits(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Derive USL/LSL as mean +/- 3*sigma per feature (6-sigma style)."""
    rows: list[dict[str, Any]] = []
    for c in feature_cols:
        x = df[c].astype(float)
        m = float(x.mean())
        s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        rows.append(
            {
                "feature": c,
                "usl": round(m + 3 * s, 4),
                "lsl": round(m - 3 * s, 4),
            },
        )
    return pd.DataFrame(rows)


def compute_capability(
    df: pd.DataFrame,
    feature_cols: list[str],
    spec_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Compute Cp, Cpk, Pp, Ppk per feature using overall sample std (Ppk-style).

    ``spec_df`` columns: feature (or Feature), usl, lsl (case-insensitive).
    """
    if spec_df is None or spec_df.empty:
        return None

    s = spec_df.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    if not {"feature", "usl", "lsl"}.issubset(s.columns):
        return None

    rows: list[dict[str, Any]] = []
    for _, row in s.iterrows():
        feat = str(row["feature"]).strip()
        if feat not in feature_cols:
            continue
        usl, lsl = float(row["usl"]), float(row["lsl"])
        if usl <= lsl:
            continue
        x = df[feat].astype(float)
        mean = float(x.mean())
        sigma = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        if sigma <= 0:
            rows.append(
                {
                    "feature": feat,
                    "cp": np.nan,
                    "cpk": np.nan,
                    "pp": np.nan,
                    "ppk": np.nan,
                    "mean": mean,
                    "sigma": sigma,
                },
            )
            continue
        cp = (usl - lsl) / (6 * sigma)
        cpu = (usl - mean) / (3 * sigma)
        cpl = (mean - lsl) / (3 * sigma)
        cpk = float(min(cpu, cpl))
        rows.append(
            {
                "feature": feat,
                "cp": float(cp),
                "cpk": cpk,
                "pp": float(cp),
                "ppk": cpk,
                "mean": mean,
                "sigma": sigma,
            },
        )

    return pd.DataFrame(rows) if rows else None


def cpk_status(cpk: float) -> str:
    if np.isnan(cpk):
        return "unknown"
    if cpk >= 1.33:
        return "good"
    if cpk >= 1.0:
        return "marginal"
    return "poor"

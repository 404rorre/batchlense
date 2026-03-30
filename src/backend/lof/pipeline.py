"""In-memory LOF scoring for DataFrames (Streamlit and APIs)."""

from __future__ import annotations

import pandas as pd

from backend.helper_modules.dataloader import DataLoader, Normalization
from backend.lof.local_outlier_factor import LOF


def cap_n_neighbors(n_neighbors: int, n_samples: int) -> tuple[int, list[str]]:
    """Return valid k and any warning messages."""
    warnings: list[str] = []
    if n_samples < 2:
        return 1, ["Fewer than 2 rows: LOF uses k=1."]
    max_k = max(1, n_samples - 1)
    k = min(n_neighbors, max_k)
    if k != n_neighbors:
        warnings.append(
            f"n_neighbors capped from {n_neighbors} to {k} (need k < n_samples).",
        )
    return k, warnings


def run_lof_on_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_df: pd.DataFrame | None = None,
    normalization: Normalization = "zscore",
    n_neighbors: int = 20,
    threshold_sigma: float = 3.0,
) -> tuple[pd.DataFrame, float, list[str]]:
    """
    Score ``df`` with LOF using reference data for threshold calibration.

    If ``reference_df`` is None, ``df`` is used for both calibration and scoring.

    Returns:
        Tuple of (result DataFrame with ``lof_score`` and ``is_outlier`` columns,
        calibrated threshold, list of warning strings).
    """
    warnings: list[str] = []
    ref = reference_df if reference_df is not None else df
    ref = ref[feature_columns].copy()
    inp = df[feature_columns].copy()

    for name, frame in (("reference", ref), ("input", inp)):
        f = frame
        if f.isna().any().any():
            raise ValueError(
                f"NaN values in {name} feature columns; drop or impute first.",
            )

    n_ref = len(ref)
    k_ref, w = cap_n_neighbors(n_neighbors, n_ref)
    warnings.extend(w)
    n_in = len(inp)
    k_in, w2 = cap_n_neighbors(n_neighbors, n_in)
    warnings.extend(w2)
    k = min(k_ref, k_in)

    ref_loader = DataLoader(normalization=normalization)
    ref_mat = ref.astype(float).values.tolist()
    ref_tensor = ref_loader.fit(data=ref_mat)

    lof_calib = LOF(n_neighbors=k, threshold=1.5)
    lof_calib.fit(ref_tensor)
    calib_scores = lof_calib.lof_scores
    assert calib_scores is not None
    threshold = float(calib_scores.mean() + threshold_sigma * calib_scores.std())

    data_loader = DataLoader(normalization=normalization)
    inp_tensor = data_loader.fit(data=inp.astype(float).values.tolist())

    lof = LOF(n_neighbors=k, threshold=threshold)
    lof.fit(inp_tensor)
    assert lof.lof_scores is not None
    assert lof.anomaly is not None

    out = df.copy()
    out["lof_score"] = lof.lof_scores
    out["is_outlier"] = lof.anomaly
    return out, threshold, warnings

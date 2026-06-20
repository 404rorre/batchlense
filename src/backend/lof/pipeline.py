"""In-memory LOF scoring for DataFrames (Streamlit and APIs)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as SkLOF

from backend.helper_modules.numpy_normalize import Normalization, RefNormalizer

_HAS_TORCH = importlib.util.find_spec("torch") is not None
"""True if PyTorch is installed (GPU extra). Public for UI/tests."""
HAS_TORCH = _HAS_TORCH


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


def _run_lof_sklearn(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_df: pd.DataFrame | None,
    normalization: Normalization,
    n_neighbors: int,
    threshold_sigma: float,
) -> tuple[pd.DataFrame, float, list[str]]:
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

    scaler = RefNormalizer(normalization)
    ref_arr = ref.astype(float).values
    inp_arr = inp.astype(float).values
    scaler.fit(ref_arr)
    ref_scaled = scaler.transform(ref_arr)
    inp_scaled = scaler.transform(inp_arr)

    lof_calib = SkLOF(
        n_neighbors=k,
        metric="euclidean",
        novelty=False,
        algorithm="brute",
    )
    lof_calib.fit(ref_scaled)
    calib_scores = -lof_calib.negative_outlier_factor_
    # Match ``ndarray.std()`` (ddof=0), same as the torch LOF calibration path.
    threshold = float(
        np.mean(calib_scores) + threshold_sigma * np.std(calib_scores),
    )

    lof = SkLOF(
        n_neighbors=k,
        metric="euclidean",
        novelty=False,
        algorithm="brute",
    )
    lof.fit(inp_scaled)
    inp_scores = -lof.negative_outlier_factor_
    is_outlier = (inp_scores > threshold).astype(np.int64)

    out = df.copy()
    out["lof_score"] = inp_scores
    out["is_outlier"] = is_outlier
    return out, float(threshold), warnings


def _run_lof_torch(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_df: pd.DataFrame | None,
    normalization: Normalization,
    n_neighbors: int,
    threshold_sigma: float,
) -> tuple[pd.DataFrame, float, list[str]]:
    from backend.helper_modules.dataloader import DataLoader
    from backend.lof.local_outlier_factor import LOF

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
    threshold = float(
        np.mean(calib_scores) + threshold_sigma * np.std(calib_scores),
    )

    inp_tensor = ref_loader.transform(inp.astype(float).values.tolist())

    lof = LOF(n_neighbors=k, threshold=threshold)
    lof.fit(inp_tensor)
    assert lof.lof_scores is not None
    assert lof.anomaly is not None

    out = df.copy()
    out["lof_score"] = lof.lof_scores
    out["is_outlier"] = lof.anomaly
    return out, threshold, warnings


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

    Uses PyTorch + custom LOF when ``torch`` is installed; otherwise scikit-learn
    LOF with the same normalization and threshold rule.

    Returns:
        Tuple of (result DataFrame with ``lof_score`` and ``is_outlier`` columns,
        calibrated threshold, list of warning strings).
    """
    if _HAS_TORCH:
        return _run_lof_torch(
            df,
            feature_columns,
            reference_df=reference_df,
            normalization=normalization,
            n_neighbors=n_neighbors,
            threshold_sigma=threshold_sigma,
        )
    return _run_lof_sklearn(
        df,
        feature_columns,
        reference_df=reference_df,
        normalization=normalization,
        n_neighbors=n_neighbors,
        threshold_sigma=threshold_sigma,
    )

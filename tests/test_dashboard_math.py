"""Unit tests for dashboard KPIs, QC math, and normalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend.helper_modules.numpy_normalize import (
    RefNormalizer,
    fit_transform_features_numpy,
)
from backend.lof.pipeline import HAS_TORCH, run_lof_on_dataframe
from frontend.components.batch_comparison import comparison_table
from frontend.components.batch_timeline import (
    baseline_feature_series,
    has_in_control_baseline,
    make_timeline_figure,
)
from frontend.components.correlation_heatmap import correlation_matrix
from frontend.components.kpi import (
    auto_spec_limits,
    basic_kpis,
    compute_capability,
    ppk_status,
)
from frontend.components.radar_chart import subset_z_for_features, z_scores_for_batch
from frontend.components.zscore_grid import make_zscore_grid_figure
from frontend.tabs.results import _lookup_spec_limits, _top_problem_features

_XBAR_LINE_COLOR = "#2E7D32"


def _xbar_y_from_figure(fig) -> float | None:
    for shape in fig.layout.shapes or []:
        if getattr(shape.line, "color", None) == _XBAR_LINE_COLOR:
            return float(shape.y0)
    return None


def test_compute_capability_pp_ppk() -> None:
    """Pp = (USL-LSL)/(6σ); Ppk = min(Ppu, Ppl)."""
    df = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
    spec = pd.DataFrame({"feature": ["x"], "usl": [10.0], "lsl": [0.0]})
    cap = compute_capability(df, ["x"], spec)
    assert cap is not None
    row = cap.iloc[0]
    mean = float(row["mean"])
    sigma = float(row["sigma"])
    assert mean == pytest.approx(5.0)
    assert sigma > 0
    pp = (10.0 - 0.0) / (6 * sigma)
    ppu = (10.0 - mean) / (3 * sigma)
    ppl = (mean - 0.0) / (3 * sigma)
    assert float(row["pp"]) == pytest.approx(pp)
    assert float(row["ppk"]) == pytest.approx(min(ppu, ppl))


def test_ppk_status_boundaries() -> None:
    assert ppk_status(float("nan")) == "unknown"
    assert ppk_status(1.33) == "good"
    assert ppk_status(1.2) == "marginal"
    assert ppk_status(1.0) == "marginal"
    assert ppk_status(0.99) == "poor"


def test_auto_spec_limits_mean_plus_minus_three_sigma() -> None:
    df = pd.DataFrame({"f": [0.0, 2.0, 4.0]})
    lim = auto_spec_limits(df, ["f"])
    m = 2.0
    s = float(pd.Series([0.0, 2.0, 4.0]).std(ddof=1))
    row = lim.loc[lim["feature"] == "f"].iloc[0]
    assert row["usl"] == pytest.approx(round(m + 3 * s, 4))
    assert row["lsl"] == pytest.approx(round(m - 3 * s, 4))


def test_basic_kpis_outliers_and_lof_range() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "is_outlier": [0, 1, 0],
            "lof_score": [1.0, 5.0, 1.2],
        },
    )
    k = basic_kpis(df, ["a", "b"])
    assert k["n_batches"] == 3
    assert k["n_outliers"] == 1
    assert k["outlier_rate_pct"] == pytest.approx(100 / 3)
    assert k["lof_min"] == 1.0
    assert k["lof_max"] == 5.0
    assert "a" in k["per_feature"]


def test_top_problem_features_ranking() -> None:
    scored = pd.DataFrame(
        {
            "a": [0.0, 0.0, 0.0, 10.0],
            "b": [0.0, 0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0],
            "is_outlier": [0, 0, 0, 1],
        },
    )
    top = _top_problem_features(scored, ["a", "b", "c"])
    assert top
    assert top[0][0] == "a"


def test_top_problem_features_no_outliers() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "is_outlier": [0, 0]})
    assert _top_problem_features(df, ["a", "b"]) == []


def test_comparison_table_delta_and_median() -> None:
    df = pd.DataFrame(
        {
            "batch": ["A", "B", "C"],
            "x": [10.0, 20.0, 30.0],
            "y": [1.0, 2.0, 3.0],
        },
    )
    comp = comparison_table(df, ["x", "y"], "batch", "B")
    med_x, med_y = 20.0, 2.0
    xr = comp.loc[comp["feature"] == "x"].iloc[0]
    assert xr["batch"] == 20.0
    assert xr["median_all"] == med_x
    assert xr["delta"] == 0.0
    assert xr["delta_pct"] == 0.0
    yr = comp.loc[comp["feature"] == "y"].iloc[0]
    assert yr["median_all"] == med_y
    assert yr["delta_pct"] == 0.0


def test_z_scores_for_batch_known_and_clip() -> None:
    # One hot row so (x - mean) / std exceeds 3 on f1; f2 constant -> z treated as 0.
    df = pd.DataFrame(
        {
            "f1": [0.0] * 19 + [100.0],
            "f2": [0.0] * 20,
        },
    )
    mask = df.index == 19
    _mean, z = z_scores_for_batch(df, ["f1", "f2"], mask)
    assert z[0] == pytest.approx(3.0)
    assert z[1] == pytest.approx(0.0)


def test_subset_z_for_features_order() -> None:
    feats = ["a", "b", "c"]
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = subset_z_for_features(feats, z, ["c", "a"])
    np.testing.assert_array_equal(out, np.array([3.0, 1.0], dtype=np.float64))


def test_zscore_grid_figure_shape() -> None:
    feats = ["a", "b", "c", "d", "e"]
    z = np.array([-3.0, 0.0, 1.5, 3.0, -1.0], dtype=np.float64)
    b = np.ones(5, dtype=np.float64)
    m = np.full(5, 2.0, dtype=np.float64)
    fig = make_zscore_grid_figure(feats, z, b, m)
    assert len(fig.data) == 1
    hm = fig.data[0]
    assert hm.type == "heatmap"
    assert hm.z.shape == (2, 3)
    z_flat = np.asarray(hm.z, dtype=float).ravel()
    finite = z_flat[np.isfinite(z_flat)]
    assert float(finite.min()) >= -3.0
    assert float(finite.max()) <= 3.0


def test_lookup_spec_limits() -> None:
    good = pd.DataFrame(
        {"feature": ["a", "b"], "usl": [10.0, 2.0], "lsl": [0.0, 0.0]},
    )
    assert _lookup_spec_limits(good, "a") == (10.0, 0.0)
    assert _lookup_spec_limits(good, "missing") == (None, None)
    assert _lookup_spec_limits(None, "a") == (None, None)
    bad_cols = pd.DataFrame({"x": [1]})
    assert _lookup_spec_limits(bad_cols, "a") == (None, None)
    inverted = pd.DataFrame({"feature": ["z"], "usl": [1.0], "lsl": [2.0]})
    assert _lookup_spec_limits(inverted, "z") == (None, None)


def test_ref_normalizer_fit_transform_matches_single_matrix() -> None:
    data = np.array([[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]])
    z1 = fit_transform_features_numpy(data, "zscore")
    n = RefNormalizer("zscore")
    n.fit(data)
    z2 = n.transform(data)
    np.testing.assert_allclose(z1, z2, atol=1e-10)


@pytest.mark.skipif(not HAS_TORCH, reason="torch optional extra")
def test_dataloader_ref_transform_matches_numpy() -> None:
    from backend.helper_modules.dataloader import DataLoader

    ref = [[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]]
    inp = [[1.0, 3.0]]
    loader = DataLoader(normalization="zscore")
    loader.fit(data=ref)
    t_np = loader.transform(inp).detach().cpu().numpy()

    n = RefNormalizer("zscore")
    n.fit(np.asarray(ref, dtype=float))
    n_np = n.transform(np.asarray(inp, dtype=float))
    np.testing.assert_allclose(t_np, n_np, atol=1e-5)


def test_correlation_matrix_identity_and_negated() -> None:
    """Spearman is 1 / -1 for perfectly monotone positive / negative pairs."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [-1, -2, -3]})
    c = correlation_matrix(df, ["a", "b", "c"])
    assert c.loc["a", "b"] == pytest.approx(1.0)
    assert c.loc["a", "c"] == pytest.approx(-1.0)


def test_histogram_kde_scaled_peak_matches_bar_peak() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=400)
    n_bins = 30
    counts, _edges = np.histogram(x, bins=n_bins)
    kde = stats.gaussian_kde(x)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    ys = kde(xs)
    ys_scaled = ys * (counts.max() / ys.max()) if ys.max() > 0 else ys
    assert float(ys_scaled.max()) == pytest.approx(float(counts.max()))


def test_run_lof_on_dataframe_adds_scores() -> None:
    df = pd.DataFrame({"a": list(range(10)), "b": list(range(10))})
    out, thr, _warns = run_lof_on_dataframe(df, ["a", "b"], n_neighbors=3)
    assert "lof_score" in out.columns
    assert "is_outlier" in out.columns
    assert len(out) == 10
    assert np.isfinite(thr)
    assert set(out["is_outlier"].unique()).issubset({0, 1})


def test_baseline_feature_series_excludes_outliers() -> None:
    df = pd.DataFrame(
        {
            "production_date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "temp": [10.0, 20.0, 100.0],
            "is_outlier": [0, 0, 1],
        },
    )
    s = baseline_feature_series(
        df,
        datetime_col="production_date",
        feature="temp",
    )
    np.testing.assert_array_equal(s.values, np.array([10.0, 20.0]))
    assert has_in_control_baseline(df, datetime_col="production_date", feature="temp")


def test_timeline_xbar_excludes_outliers() -> None:
    df = pd.DataFrame(
        {
            "production_date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "temp": [10.0, 20.0, 100.0],
            "is_outlier": [0, 0, 1],
        },
    )
    fig = make_timeline_figure(
        df,
        datetime_col="production_date",
        feature="temp",
        resample="day",
    )
    assert _xbar_y_from_figure(fig) == pytest.approx(15.0)


def test_timeline_xbar_hidden_all_outliers() -> None:
    df = pd.DataFrame(
        {
            "production_date": pd.date_range("2026-01-01", periods=2, freq="D"),
            "temp": [10.0, 20.0],
            "is_outlier": [1, 1],
        },
    )
    assert not has_in_control_baseline(
        df,
        datetime_col="production_date",
        feature="temp",
    )
    fig = make_timeline_figure(
        df,
        datetime_col="production_date",
        feature="temp",
        resample="day",
    )
    assert _xbar_y_from_figure(fig) is None


def test_hourly_limits_use_in_control_when_present() -> None:
    df = pd.DataFrame(
        {
            "production_date": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temp": [10.0, 20.0, 30.0, 100.0],
            "is_outlier": [0, 0, 0, 1],
        },
    )
    in_control = baseline_feature_series(
        df,
        datetime_col="production_date",
        feature="temp",
        in_control_only=True,
    )
    all_rows = baseline_feature_series(
        df,
        datetime_col="production_date",
        feature="temp",
        in_control_only=False,
    )
    assert float(in_control.mean()) == pytest.approx(20.0)
    assert float(all_rows.mean()) == pytest.approx(40.0)

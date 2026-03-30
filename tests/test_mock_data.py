"""Tests for synthetic demo batch data."""

from __future__ import annotations

import pandas as pd

from frontend.mock_data import FEATURE_COLUMNS, generate_mock_batches


def test_generate_mock_batches_shape_and_columns() -> None:
    df, mask = generate_mock_batches(n_batches=120, seed=0)
    assert len(df) == 120
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert "batch_number" in df.columns
    assert "production_date" in df.columns
    assert mask.dtype == bool
    assert mask.sum() >= 1


def test_production_dates_span_about_a_week() -> None:
    df, _ = generate_mock_batches(n_batches=120, seed=1)
    t0 = pd.to_datetime(df["production_date"]).min()
    t1 = pd.to_datetime(df["production_date"]).max()
    delta = (t1 - t0).total_seconds()
    assert 5 * 86400 < delta < 8 * 86400


def test_outlier_fraction_in_expected_range() -> None:
    for seed in range(20):
        df, mask = generate_mock_batches(n_batches=200, seed=seed)
        frac = mask.sum() / len(df)
        assert 0.07 <= frac <= 0.15

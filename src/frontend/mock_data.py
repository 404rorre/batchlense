"""Synthetic chemical batch telemetry for demo mode (no Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SPECS: dict[str, tuple[float, float]] = {
    "reactor_temperature": (182.0, 2.0),
    "reactor_pressure": (4.2, 0.15),
    "pH": (7.1, 0.2),
    "viscosity": (340.0, 12.0),
    "catalyst_concentration": (0.85, 0.03),
    "yield_pct": (94.5, 1.2),
}

FEATURE_COLUMNS: list[str] = list(FEATURE_SPECS.keys())


def generate_mock_batches(
    n_batches: int = 120,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a week of batch rows with realistic process columns.

    A random fraction between 7% and 15% of rows are perturbed on 1–3 features
    by roughly 2.5–4σ so LOF can flag them. Returns ``(df, is_synthetic_outlier)``.
    """
    rng = np.random.default_rng(seed)
    low = max(1, int(np.ceil(n_batches * 0.07)))
    high = min(n_batches, int(np.floor(n_batches * 0.15)))
    if high < low:
        low = high = max(1, min(n_batches, low))
    n_out = int(rng.integers(low, high + 1))

    start = pd.Timestamp("2026-03-23 06:00:00")
    end = pd.Timestamp("2026-03-29 22:00:00")
    span = (end - start).total_seconds()
    offsets = np.sort(rng.uniform(0.0, 1.0, size=n_batches))
    jitter_sec = rng.integers(-1800, 1801, size=n_batches)
    prod_times = start + pd.to_timedelta(offsets * span + jitter_sec, unit="s")

    rows: dict[str, np.ndarray] = {
        "batch_number": np.array(
            [f"B-2026-{i + 1:03d}" for i in range(n_batches)],
            dtype=object,
        ),
        "production_date": prod_times,
    }

    for col, (mean, std) in FEATURE_SPECS.items():
        rows[col] = np.round(rng.normal(mean, std, size=n_batches), 2)

    df = pd.DataFrame(rows)
    df["production_date"] = pd.to_datetime(df["production_date"])

    outlier_mask = np.zeros(n_batches, dtype=bool)
    out_idx = rng.choice(n_batches, size=min(n_out, n_batches), replace=False)
    outlier_mask[out_idx] = True

    for i in out_idx:
        n_feat = int(rng.integers(1, 4))
        pick = rng.choice(FEATURE_COLUMNS, size=n_feat, replace=False)
        direction = rng.choice([-1.0, 1.0])
        mult = float(rng.uniform(2.5, 4.0))
        for col in pick:
            mean, std = FEATURE_SPECS[col]
            df.loc[i, col] = round(
                float(df.loc[i, col]) + direction * mult * std,
                2,
            )

    return df, pd.Series(outlier_mask, name="is_synthetic_outlier")

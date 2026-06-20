"""Feature normalization without PyTorch (matches ``DataLoader`` numerics)."""

from __future__ import annotations

from typing import Literal

import numpy as np

Normalization = Literal["minmax", "zscore", "none", ""]


def fit_transform_features_numpy(
    data: np.ndarray,
    normalization: Normalization,
) -> np.ndarray:
    """
    Fit normalization on ``data`` and return the transformed matrix.

    Same semantics as ``DataLoader.fit`` when ``data`` is the only matrix.
    """
    x = np.asarray(data, dtype=np.float64)
    if normalization == "minmax":
        fit_min = x.min(axis=0)
        fit_max = x.max(axis=0)
        denom = fit_max - fit_min
        denom = np.where(denom == 0, 1.0, denom)
        return np.asarray((x - fit_min) / denom, dtype=np.float64)
    if normalization == "zscore":
        fit_mean = x.mean(axis=0)
        # Match ``torch.Tensor.std(dim=0)`` (unbiased / ddof=1).
        fit_std = x.std(axis=0, ddof=1)
        fit_std = np.where(
            ~np.isfinite(fit_std) | (fit_std == 0),
            1.0,
            fit_std,
        )
        return np.asarray((x - fit_mean) / (fit_std + 1e-10), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


class RefNormalizer:
    """Fit on reference rows, then transform other matrices (like ``DataLoader``)."""

    def __init__(self, normalization: Normalization) -> None:
        self.normalization = normalization
        self._fit_min: np.ndarray | None = None
        self._fit_max: np.ndarray | None = None
        self._fit_mean: np.ndarray | None = None
        self._fit_std: np.ndarray | None = None

    def fit(self, ref: np.ndarray) -> None:
        x = np.asarray(ref, dtype=np.float64)
        if self.normalization == "minmax":
            self._fit_min = x.min(axis=0)
            self._fit_max = x.max(axis=0)
        elif self.normalization == "zscore":
            self._fit_mean = x.mean(axis=0)
            self._fit_std = x.std(axis=0, ddof=1)
            self._fit_std = np.where(
                ~np.isfinite(self._fit_std) | (self._fit_std == 0),
                1.0,
                self._fit_std,
            )
        else:
            self._fit_min = self._fit_max = self._fit_mean = self._fit_std = None

    def transform(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=np.float64)
        if self.normalization == "minmax":
            assert self._fit_min is not None and self._fit_max is not None
            denom = self._fit_max - self._fit_min
            denom = np.where(denom == 0, 1.0, denom)
            return np.asarray((x - self._fit_min) / denom, dtype=np.float64)
        if self.normalization == "zscore":
            assert self._fit_mean is not None and self._fit_std is not None
            return np.asarray(
                (x - self._fit_mean) / (self._fit_std + 1e-10),
                dtype=np.float64,
            )
        return np.asarray(x, dtype=np.float64)

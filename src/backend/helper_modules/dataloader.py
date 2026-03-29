"""Load tabular data and normalize features as PyTorch tensors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, tensor

from .tools import timer_func

logger = logging.getLogger(__name__)

Normalization = Literal["minmax", "zscore", "none", ""]


class DataLoader:
    """
    A data loader class for handling dataset splitting and normalization.
    """

    normalization: str
    split: tuple[float, float, float]

    def __init__(
        self,
        split: tuple[int, int, int] = (80, 10, 10),
        normalization: Normalization = "minmax",
    ) -> None:
        """
        Initialize the data loader with splitting and normalization options.

        Args:
            split: Percentages for train, validation, and test (must sum to 100).
            normalization: One of ``minmax``, ``zscore``, ``none``, or ``""``.
        """
        self.normalization = normalization
        self.split = cast(tuple[float, float, float], split)
        self._sanitization()

    def _sanitization(self) -> None:
        """Validate split and normalization options."""
        if isinstance(self.split, tuple):
            total = sum(self.split)
            if total == 100:
                self.split = cast(
                    tuple[float, float, float],
                    tuple(n / 100 for n in self.split),
                )
            elif total != 1:
                raise ValueError("Split is not 100% in total.")
            else:
                self.split = cast(
                    tuple[float, float, float],
                    tuple(float(x) for x in self.split),
                )

        norm_methods: list[str | None] = ["minmax", "zscore", "none", None, ""]
        if self.normalization not in norm_methods:
            raise ValueError("Normalization is not listed in available methods.")

    @timer_func
    def fit_autoencoder(
        self,
        f_path: str | Path | None = None,
        data: list[Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Load data, normalize, and split for autoencoder training.

        Returns:
            Tuple of (train, validation, test) tensors.
        """
        self.f_path = f_path
        self.data = data
        self._load_data()
        self._normalize_data()

        n_samples = self.x.shape[0]
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * self.split[0])
        val_end = int(n_samples * (self.split[0] + self.split[1]))

        train_data = self.x[indices[:train_end]]
        val_data = self.x[indices[train_end:val_end]]
        test_data = self.x[indices[val_end:]]

        return train_data, val_data, test_data

    def fit(
        self,
        f_path: str | Path | None = None,
        data: list[Any] | None = None,
    ) -> Tensor:
        """
        Load and normalize data from a CSV path or in-memory list.

        Returns:
            Feature matrix as a float tensor.
        """
        self.f_path = f_path
        self.data = data
        self._load_data()
        self._normalize_data()
        return self.x

    def _load_data(self) -> None:
        """Populate ``self.x`` from list input or CSV file."""
        if isinstance(self.data, list):
            self.x = tensor(self.data).float()
            return

        if self.f_path is None:
            raise ValueError("Either ``data`` (list) or ``f_path`` must be provided.")

        path = Path(self.f_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")

        self.x_csv = pd.read_csv(path)
        self.x = tensor(self.x_csv.to_numpy()).float()

    def _normalize_data(self) -> None:
        """Apply configured normalization to ``self.x``."""
        if self.normalization == "minmax":
            self._norm_minmax()
        elif self.normalization == "zscore":
            self._norm_z()
        else:
            # "none", None, or ""
            logger.debug("No normalization applied")

        logger.debug("Data shape: %s (n=%s)", self.x.shape, self.x.shape[0])

    def _norm_minmax(self) -> None:
        """Scale each feature to [0, 1]."""
        x_min = self.x.min(dim=0).values
        x_max = self.x.max(dim=0).values
        denom = x_max - x_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        self.x = (self.x - x_min) / (denom + 1e-10)

    def _norm_z(self) -> None:
        """Z-score each feature (mean 0, std 1)."""
        std = self.x.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.x = (self.x - self.x.mean(dim=0)) / (std + 1e-10)

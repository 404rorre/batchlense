"""PyTorch Local Outlier Factor (LOF) for anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, tensor

from backend.helper_modules.tools import timer_func

# During ``fit``, ``lof_scores`` is a tensor until the final step converts to NumPy.
ScoreArray = np.ndarray | Tensor


class LOF:
    """
    Local Outlier Factor: scores reflect how isolated each point is vs. its k neighbors.
    """

    n_neighbors: int
    k: int
    threshold: float
    lof_scores: ScoreArray | None
    anomaly: np.ndarray | None
    device: torch.device

    def __init__(
        self,
        n_neighbors: int = 20,
        threshold: float = 1.5,
        device: torch.device | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.k = self.n_neighbors
        self.threshold = threshold
        self.lof_scores = None
        self.anomaly = None
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_sanitation()

    def __repr__(self) -> str:
        return f"LOF(n_neighbors={self.n_neighbors!r}, threshold={self.threshold!r})"

    @timer_func
    def fit(self, x: Tensor | np.ndarray | list[list[float]]) -> None:
        """
        Fit LOF on ``x`` (tensor, ndarray, or nested list).

        After ``fit``, scores and labels are NumPy arrays on CPU.
        """
        if isinstance(x, Tensor):
            self.x = x.to(dtype=torch.float32, device=self.device)
        else:
            self.x = tensor(x, dtype=torch.float32, device=self.device)
        self._local_outlier_factor()
        scores_t = self.lof_scores
        assert isinstance(scores_t, Tensor)
        self.anomaly = self._binary_classifier(scores_t)
        self.lof_scores = scores_t.squeeze(1).detach().cpu().numpy()

    def _binary_classifier(self, scores: Tensor) -> np.ndarray:
        mask = torch.where(scores > self.threshold, 1, 0)
        return mask.squeeze(1).detach().cpu().numpy()

    def _local_outlier_factor(self) -> None:
        """Pairwise distances, reachability, LRD, and LOF on ``self.device``."""
        self.point_dist = torch.cdist(self.x, self.x)
        self.topk = self.point_dist.topk(
            k=self.n_neighbors + 1,
            dim=1,
            largest=False,
            sorted=True,
        )
        k_idx = tensor([self.k - 1], device=self.device, dtype=torch.long)
        self.kdist = torch.index_select(
            input=self.topk.values[:, 1:],
            dim=1,
            index=k_idx,
        ).squeeze(1)
        kdist_comp = torch.gather(
            self.kdist.expand(self.point_dist.shape),
            1,
            self.topk.indices[:, 1:],
        )
        self.rd = torch.max(self.topk.values[:, 1:], kdist_comp)
        self.lrd = (1 / (self.rd.sum(dim=1) / (self.n_neighbors + 1e-10))).unsqueeze(1)
        lrd_exp = self.lrd.squeeze(1).expand(self.point_dist.shape)
        self.lof_scores = (
            torch.gather(lrd_exp, 1, self.topk.indices[:, 1:]).sum(dim=1).unsqueeze(1)
            / (self.n_neighbors)
        ) / self.lrd

    def _input_sanitation(self) -> None:
        if not isinstance(self.n_neighbors, (int, float)):
            raise TypeError("Attribute 'n_neighbors' needs to be an integer.")
        if not float(self.n_neighbors).is_integer():
            msg = "Attribute 'n_neighbors' must be an integer (float detected)."
            raise ValueError(msg)
        if not isinstance(self.k, (int, float)):
            raise TypeError("Attribute 'k' needs to be an integer.")
        if not float(self.k).is_integer():
            msg = "Attribute 'k' must be an integer (float detected)."
            raise ValueError(msg)
        if not isinstance(self.threshold, (int, float)):
            raise TypeError("Attribute 'threshold' needs to be a float.")
        self.threshold = float(self.threshold)

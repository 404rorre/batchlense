"""Tests comparing custom LOF to scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.neighbors import LocalOutlierFactor as SkLOF

from backend.helper_modules.dataloader import DataLoader
from backend.lof.local_outlier_factor import LOF


@pytest.fixture
def mixed_cluster_example() -> list[list[int]]:
    return [
        [1, 2],
        [2, 3],
        [3, 4],
        [100, 200],
        [2, 2],
        [2, 6],
        [0, 1],
        [3, 6],
        [7, 8],
        [3, 2],
        [1, 1],
    ]


def test_lof_scores_match_sklearn(mixed_cluster_example: list[list[int]]) -> None:
    data_test = DataLoader()
    x_tensor = data_test.fit(data=mixed_cluster_example)
    n_neighbors = 4
    x_sklearn = x_tensor.detach().cpu().numpy()

    lof_sklearn = SkLOF(n_neighbors=n_neighbors, metric="euclidean", algorithm="brute")
    lof_sklearn.fit(x_sklearn)
    sklof_scores = -lof_sklearn.negative_outlier_factor_

    lof = LOF(n_neighbors=n_neighbors)
    lof.fit(x_tensor)
    assert lof.lof_scores is not None
    np.testing.assert_allclose(
        sklof_scores,
        lof.lof_scores,
        atol=1e-5,
        err_msg="Custom LOF scores diverge from sklearn",
    )


def test_lof_with_tight_cluster_near_one() -> None:
    rng = np.random.default_rng(42)
    x_list = (rng.normal(0.0, 0.01, size=(40, 2))).tolist()
    data = DataLoader(normalization="none")
    x_tensor = data.fit(data=x_list)
    lof = LOF(n_neighbors=10)
    lof.fit(x_tensor)
    assert lof.lof_scores is not None
    # Inliers in a dense cluster should have LOF close to 1 on average.
    assert float(np.mean(lof.lof_scores)) == pytest.approx(1.0, abs=0.25)


def test_lof_invalid_n_neighbors_raises() -> None:
    with pytest.raises(TypeError):
        LOF(n_neighbors="not_an_int")  # type: ignore[arg-type]


def test_lof_repr() -> None:
    lof = LOF(n_neighbors=5, threshold=2.0)
    assert repr(lof) == "LOF(n_neighbors=5, threshold=2.0)"


def test_lof_uses_cpu_when_explicit() -> None:
    device = torch.device("cpu")
    lof = LOF(n_neighbors=3, device=device)
    lof.fit([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert lof.lof_scores is not None
    assert lof.lof_scores.shape == (4,)

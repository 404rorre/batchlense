"""2D projections for visualization (PCA, t-SNE, UMAP) with fixed random seed."""

from __future__ import annotations

from typing import Literal

import numpy as np

RANDOM_STATE = 42

Method = Literal["pca", "tsne", "umap"]


def project_2d(
    X: np.ndarray,
    method: Method,
) -> tuple[np.ndarray, list[str]]:
    """
    Project feature matrix ``X`` (n_samples, n_features) to 2D.

    Returns (coords, warnings).
    """
    warnings: list[str] = []
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return np.zeros((max(n, 1), 2)), ["Not enough rows for 2D projection."]

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(2, X.shape[1]), random_state=RANDOM_STATE)
        z = pca.fit_transform(X)
        if z.shape[1] == 1:
            z = np.column_stack([z, np.zeros(n)])
        return z.astype(np.float64), warnings

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, n - 1))
        if perplexity < 30:
            warnings.append(
                f"t-SNE perplexity set to {perplexity} (capped for sample size).",
            )
        tsne = TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(X).astype(np.float64), warnings

    import umap

    n_neighbors = min(15, max(2, n - 1))
    z = umap.UMAP(
        n_components=2,
        random_state=RANDOM_STATE,
        n_neighbors=n_neighbors,
        min_dist=0.1,
    ).fit_transform(X)
    return z.astype(np.float64), warnings

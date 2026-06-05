"""PCA-based dimensionality reduction and visualization."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(feature_df: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, PCA]:
    """Reduce feature space to 2 principal components."""
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(feature_df)
    pca_df = pd.DataFrame(
        components, columns=["PC1", "PC2"], index=feature_df.index
    )
    return pca_df, pca


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    cluster_labels,
    output_path: str,
) -> None:
    """Save a PCA cluster visualization plot."""
    figure, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=cluster_labels,
        cmap="tab10",
        s=60,
        alpha=0.8,
        edgecolors="w",
        linewidth=0.5,
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Cluster Visualization")
    ax.grid(alpha=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

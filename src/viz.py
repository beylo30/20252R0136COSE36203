from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def compare_augmentation(data_augmentation, image: np.ndarray, save_path: Path | None = None) -> None:
    import tensorflow as tf

    augmented_image = data_augmentation(tf.expand_dims(image, 0), training=True)[0].numpy()

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title("Augmented")
    plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def pca_3d_plot(features: np.ndarray, labels: np.ndarray, save_path: Path | None = None) -> None:
    """3D PCA scatter plot. Uses Plotly if available, else matplotlib."""
    pca = PCA(n_components=3, random_state=42)
    feat_3d = pca.fit_transform(features)

    # Try Plotly first
    try:
        import plotly.express as px  # type: ignore

        fig = px.scatter_3d(
            x=feat_3d[:, 0],
            y=feat_3d[:, 1],
            z=feat_3d[:, 2],
            color=labels.astype(str),
            title="3D PCA of CNN Features (CIFAR-10)",
            labels={"color": "Class"},
        )
        if save_path is not None:
            # Writes an HTML file if kaleido isn't installed.
            save_path = save_path.with_suffix(".html")
            fig.write_html(str(save_path))
        fig.show()
        return
    except Exception:
        pass

    # Matplotlib fallback
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(feat_3d[:, 0], feat_3d[:, 1], feat_3d[:, 2], c=labels, s=5)
    ax.set_title("3D PCA of CNN Features (CIFAR-10)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

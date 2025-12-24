from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import requests


@dataclass(frozen=True)
class Prediction:
    top_class: str
    top_prob: float
    probs: np.ndarray
    local_path: str  # where the image was saved


def _download_from_url(url: str, prefix: str = "external") -> str:
    """
    Download URL to a unique cached file using a browser-like User-Agent.
    Fixes:
    - caching bug (unique name per URL)
    - 403 forbidden from some websites
    """
    cache_dir = Path("outputs") / "url_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    local_path = cache_dir / f"{prefix}_{url_hash}.jpg"

    # if already downloaded, reuse
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=30, stream=True)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return str(local_path)


def predict_image_from_url(
    model: tf.keras.Model,
    url: str,
    class_names: List[str],
    target_size: Tuple[int, int] = (32, 32),
) -> Prediction:
    local_path = _download_from_url(url, prefix="external_img")

    # model input (32x32)
    img_32 = keras.utils.load_img(local_path, target_size=target_size)
    x = keras.utils.img_to_array(img_32).astype("float32") / 255.0
    x = tf.expand_dims(x, 0)  # (1, 32, 32, 3)

    raw = model.predict(x, verbose=0)[0]
    raw = np.asarray(raw)

    # probabilities safely (softmax only if needed)
    if (raw.min() >= 0.0) and (raw.max() <= 1.0) and np.isclose(raw.sum(), 1.0, atol=1e-3):
        probs = raw
    else:
        probs = tf.nn.softmax(raw).numpy()

    top = int(np.argmax(probs))
    return Prediction(
        top_class=class_names[top],
        top_prob=float(probs[top]),
        probs=probs,
        local_path=local_path,
    )


def show_downloaded_and_model_input(local_path: str, title: str = "") -> None:
    """
    Show:
    - left: downloaded image (original size)
    - right: model input (32x32)
    """
    img_full = keras.utils.load_img(local_path)
    img_32 = keras.utils.load_img(local_path, target_size=(32, 32))

    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img_full)
    plt.title("Downloaded image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_32)
    plt.title(title if title else "Model input (32x32)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def show_image_from_url(url: str, title: str) -> None:
    local_path = _download_from_url(url, prefix="external_img_vis")
    show_downloaded_and_model_input(local_path, title=title)

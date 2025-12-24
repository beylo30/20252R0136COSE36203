from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10


@dataclass(frozen=True)
class Cifar10Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List[str]


def load_cifar10(val_split: float = 0.2, seed: int = 42) -> Cifar10Data:
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

    classes = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]

    y_train_full = y_train_full.reshape(-1).astype(np.int64)
    y_test = y_test.reshape(-1).astype(np.int64)

    # Train/val split from official training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_split,
        random_state=seed,
        stratify=y_train_full,
    )

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return Cifar10Data(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
    )


def make_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(20000, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(autotune)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(autotune)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(autotune)
    return train_ds, val_ds, test_ds

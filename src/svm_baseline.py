from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class SvmResult:
    best_params: Dict[str, float]
    val_accuracy: float
    test_accuracy: float


def extract_cnn_features(
    model: tf.keras.Model,
    train_ds_noshuf: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_extractor = tf.keras.Model(model.input, model.get_layer("features").output)

    train_features = feature_extractor.predict(train_ds_noshuf, verbose=0)
    val_features = feature_extractor.predict(val_ds, verbose=0)
    test_features = feature_extractor.predict(test_ds, verbose=0)

    train_features = train_features.reshape(train_features.shape[0], -1)
    val_features = val_features.reshape(val_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)
    return train_features, val_features, test_features


def svm_on_features(
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
    test_features: np.ndarray,
    y_test: np.ndarray,
    c_grid: Iterable[float] = (0.1, 1, 3, 10),
) -> SvmResult:
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(dual=False, max_iter=5000)),
        ]
    )

    param_grid = {"svm__C": list(c_grid)}
    grid = GridSearchCV(svm_pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(train_features, y_train)

    best = grid.best_estimator_

    val_pred = best.predict(val_features)
    val_acc = float(accuracy_score(y_val, val_pred))

    test_pred = best.predict(test_features)
    test_acc = float(accuracy_score(y_test, test_pred))

    return SvmResult(
        best_params=grid.best_params_,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
    )

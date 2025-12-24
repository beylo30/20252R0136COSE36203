from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[float, float]:
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return float(test_loss), float(test_acc)


def plot_training_curves(history: tf.keras.callbacks.History, save_path: Path | None = None) -> None:
    hist = history.history
    epochs_r = range(1, len(hist["accuracy"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_r, hist["accuracy"], label="Train Acc")
    plt.plot(epochs_r, hist["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_r, hist["loss"], label="Train Loss")
    plt.plot(epochs_r, hist["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def classification_report_and_cm(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_test: np.ndarray,
    class_names: List[str],
    save_path: Path | None = None,
) -> np.ndarray:
    y_pred_logits = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_logits, axis=1)

    print("\nClassification Report (TEST):")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, save_path=save_path)
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix (TEST)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

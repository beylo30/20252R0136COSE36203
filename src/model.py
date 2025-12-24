from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_data_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def conv_bn_lrelu(x: tf.Tensor, filters: int, k: int = 3, s: int = 1, l2=None) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        k,
        strides=s,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    return x


def residual_block(x: tf.Tensor, filters: int, downsample: bool = False, l2=None) -> tf.Tensor:
    stride = 2 if downsample else 1
    shortcut = x

    x = conv_bn_lrelu(x, filters, k=3, s=stride, l2=l2)
    x = layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2,
    )(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions for residual add
    if downsample or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.1)(x)
    return x


def build_resnet_small(num_classes: int = 10, weight_decay: float = 1e-4) -> keras.Model:
    l2 = regularizers.l2(weight_decay)
    data_aug = build_data_augmentation()

    inputs = keras.Input(shape=(32, 32, 3))
    x = data_aug(inputs)

    # Stem
    x = conv_bn_lrelu(x, 32, l2=l2)
    x = conv_bn_lrelu(x, 32, l2=l2)

    # Stages
    x = residual_block(x, 32, l2=l2)
    x = residual_block(x, 32, l2=l2)

    x = residual_block(x, 64, downsample=True, l2=l2)
    x = residual_block(x, 64, l2=l2)

    x = residual_block(x, 128, downsample=True, l2=l2)
    x = residual_block(x, 128, l2=l2)

    x = residual_block(x, 256, downsample=True, l2=l2)
    x = residual_block(x, 256, l2=l2)

    x = layers.GlobalAveragePooling2D(name="features")(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, logits, name="CIFAR10_ResNetSmall")


def compile_with_cosine_sgd(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    epochs: int,
    initial_lr: float = 0.1,
    momentum: float = 0.9,
) -> None:
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=steps_per_epoch * epochs,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

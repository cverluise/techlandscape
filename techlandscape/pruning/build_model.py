from keras import models
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)
from techlandscape.utils import get_config
from pathlib import Path
from typing import Tuple

"""
    V0 based on https://developers.google.com/machine-learning/guides/text-classification/
    Other useful resources https://realpython.com/python-keras-text-classification/
    On CNN http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform
    -text-classification-with-word-embeddings/
"""


# TODO: implement early stopping in model_*


def build_mlp(config: Path, input_shape) -> models.Sequential:
    """
    Return a Multi layer perceptron Keras model
    """
    # TODO solve input shape - should be declared on config or infered from the data
    #  input_shape is x_train.shape[1:]

    cfg = get_config(config)
    # Init model
    model = models.Sequential()
    model.add(Dropout(rate=cfg["model"]["dropout_rate"], input_shape=input_shape))

    # Add mlp layers
    for _ in range(cfg["model"]["layers"] - 1):
        model.add(Dense(units=cfg["model"]["units"], activation="relu"))
        model.add(Dropout(rate=cfg["model"]["dropout_rate"]))

    # Binary classification + output ~ proba
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def build_cnn(
    config: Path, input_shape: Tuple, num_features: int, embedding_matrix: dict = None
) -> models.Sequential:
    """
    Return a CNN model with <blocks> Convolution-Pooling pair layers.
    """
    # TODO input_shape and num_features should be inferred or declared
    #  input_shape: shape of input to the model.
    #  num_features: number of words (number of columns in embedding input).
    cfg = get_config(config)

    model = models.Sequential()
    if cfg["model"]["use_pretrained_embedding"]:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=cfg["model"]["embedding_dim"],
                input_length=input_shape[0],
                weights=[embedding_matrix],
                trainable=cfg["model"]["is_embedding_trainable"],
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=cfg["model"]["embedding_dim"],
                input_length=input_shape[0],
            )
        )
    for _ in range(cfg["model"]["blocks"] - 1):
        model.add(Dropout(rate=cfg["model"]["dropout_rate"]))
        model.add(
            Conv1D(
                filters=cfg["model"]["filters"],
                kernel_size=cfg["model"]["kernel_size"],
                activation="relu",
                bias_initializer="random_uniform",
                padding="same",
            )
        )
        model.add(MaxPooling1D(pool_size=cfg["model"]["pool_size"]))

    model.add(
        Conv1D(
            filters=cfg["model"]["filters"] * 2,
            kernel_size=cfg["model"]["kernel_size"],
            activation="relu",
            bias_initializer="random_uniform",
            padding="same",
        )
    )
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=cfg["model"]["dropout_rate"]))
    model.add(Dense(1, activation="sigmoid"))
    return model

from keras import models
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)

"""
    V0 based on https://developers.google.com/machine-learning/guides/text-classification/
    Other useful resources https://realpython.com/python-keras-text-classification/
    On CNN http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform
    -text-classification-with-word-embeddings/
"""


# TODO: implement early stopping in model_*


def build_mlp(layers, units, dropout_rate, input_shape):
    """
    Return a MLP with <layers> layers.
    :param layers: int, number of hidden layers -1. If 1, then no hidden layer.
    :param units: int, number of units per hidden layer.
    :param dropout_rate: float [0,1], fraction of the input units to drop.
    :param input_shape: tuple, shape of the input.
    :return: keras.models.Sequential, MLP model
    """
    # Init model
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    # Add mlp layers
    for _ in range(layers - 1):
        model.add(Dense(units=units, activation="relu"))
        model.add(Dropout(rate=dropout_rate))

    # Binary classification + output ~ proba
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def build_cnn(
    blocks,
    filters,
    kernel_size,
    embedding_dim,
    dropout_rate,
    pool_size,
    input_shape,
    num_features,
    use_pretrained_embedding=False,
    is_embedding_trainable=False,
    embedding_matrix=None,
):
    """
    Return a CNN model with <blocks> Convolution-Polling pairs.
    :param blocks: int, number of Convolution-Pooling pairs.
    :param filters: int, the dimensionality of the output space (i.e. the number of output
    filters in the convolution).
    :param kernel_size: int, length of the 1D convolution window.
    :param embedding_dim: int, dimension of the embedding vectors (reco between 50 and 200).
    :param dropout_rate: float, percentage of input to drop at Dropout Layer.
    :param pool_size: int, factor by which to downscale input at MaxPooling layer.
    :param input_shape: tuple, shape of input to the model.
    :param num_features: int, number of words (number of columns in embedding input).
    :param use_pretrained_embedding: bool, true if pre-trained .
    :param is_embedding_trainable: bool, true if embedding trainable.
    :param embedding_matrix: dict, dictionary with embedding coefficients.
    :return: keras.models.Sequential, CNN model
    """
    model = models.Sequential()
    if use_pretrained_embedding:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=embedding_dim,
                input_length=input_shape[0],
                weights=[embedding_matrix],
                trainable=is_embedding_trainable,
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=embedding_dim,
                input_length=input_shape[0],
            )
        )
    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                bias_initializer="random_uniform",
                padding="same",
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(
        Conv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            activation="relu",
            bias_initializer="random_uniform",
            padding="same",
        )
    )
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    return model

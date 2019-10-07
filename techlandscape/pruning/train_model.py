from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from techlandscape.pruning import vectorize_text, build_model


# TODO? save the model
# TODO? 'f_score' in metrics (model.compile). See custom metrics https://keras.io/metrics/
# TODO? implement EarlyStopping on f_score


def train_mlp(
    texts_train,
    texts_test,
    y_train,
    y_test,
    learning_rate=1e-3,
    epochs=100,
    batch_size=64,
    layers=2,
    units=64,
    dropout_rate=0.2,
):
    """
    Return the trained MLP model.
    :param texts_train: List[str], list of training set texts.
    :param texts_test: List[str], list of test set texts.
    :param y_train: arr, training set labels.
    :param y_test: arr, test set labels.
    :param learning_rate: float, learning rate.
    :param epochs: int, number of epochs to train the model.
    :param batch_size: int, number of samples per gradient update.
    :param layers: int, number of hidden layers -1. If 1, then no hidden layer.
    :param units: int, number of units per hidden layer.
    :param dropout_rate: float [0,1], fraction of the input units to drop.
    :return: fitted model, model history
    """

    # Vectorize texts.
    x_train, x_test = vectorize_text.get_ngram(
        texts_train, y_train, texts_test
    )

    # Create model instance.
    model = build_model.build_mlp(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
    )
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [EarlyStopping(monitor="val_loss", patience=2)]

    # Train and validate model.
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size,
    )

    # Print results.
    history = history.history
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_accuracy"][-1], loss=history["val_loss"][-1]
        )
    )

    return model, history


def train_cnn(
    texts_train,
    texts_test,
    y_train,
    y_test,
    learning_rate=1e-3,
    epochs=100,
    batch_size=64,
    blocks=2,
    filters=64,
    dropout_rate=0.2,
    embedding_dim=200,
    kernel_size=3,
    pool_size=3,
):
    """

    :param texts_train: List[str], list of training set texts.
    :param texts_test: List[str], list of test set texts.
    :param y_train: arr, training set labels.
    :param y_test: arr, test set labels.
    :param learning_rate: float, learning rate.
    :param epochs: int, number of epochs to train the model.
    :param batch_size: int, number of samples per gradient update.
    :param blocks: int, number of Convolution-Pooling pairs.
    :param filters: int, the dimensionality of the output space (i.e. the number of output
    filters in the convolution).
    :param dropout_rate: float, percentage of input to drop at Dropout Layer.
    :param embedding_dim: int, dimension of the embedding vectors (reco between 50 and 200).
    :param kernel_size: int, length of the 1D convolution window.
    :param pool_size: int, factor by which to downscale input at MaxPooling layer.
    :return: fitted model, model history
    """
    # Vectorize texts.
    x_train, x_test, word_index = vectorize_text.get_sequence(
        texts_train, texts_test
    )
    num_features = min(len(word_index) + 1, vectorize_text.TOP_K)

    # Buid models.
    model = build_model.build_cnn(
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        pool_size=pool_size,
        input_shape=x_train.shape[1:],
        num_features=num_features,
    )

    # Compile model with learning parameters.
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [EarlyStopping(monitor="val_loss", patience=2)]

    # Train and validate model.
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size,
    )

    # Print results.
    history = history.history
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_accuracy"][-1], loss=history["val_loss"][-1]
        )
    )

    return model, history

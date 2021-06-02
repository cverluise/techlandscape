import numpy as np
import json
from pathlib import Path
from keras.preprocessing import sequence, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from keras import models
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from techlandscape.utils import get_config, ok, not_ok
from techlandscape.exception import UnknownModel, UNKNOWN_MODEL_MSG
from techlandscape.enumerators import SupportedModels
import typer


class DataLoader:
    """
    Load data (train, test)

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import DataLoader
        data_loader = DataLoader("configs/model_cnn.yaml")
        data_loader.load_data()

        # check examples
        data_loader.text_train
        ```
    """

    text_train = None
    text_test = None
    y_train = None
    y_test = None

    def __init__(self, config: Path):
        self.cfg = get_config(config)
        self.train_path = Path(self.cfg["data"]["train"])
        self.test_path = Path(self.cfg["data"]["test"])

    @staticmethod
    def _get_data(path: Path, var: str):
        return [
            json.loads(line)[var] for line in path.open("r").read().split("\n") if line
        ]

    def load_data(self):
        """Load data. Expect a jsonl file where each row at least two fields: 'text' and 'is_seed'."""
        if any(
            map(
                lambda x: x is None,
                [self.text_train, self.text_test, self.y_train, self.y_test],
            )
        ):
            self.text_train = self._get_data(self.train_path, "text")
            self.text_test = self._get_data(self.test_path, "text")
            self.y_train = np.array(self._get_data(self.train_path, "is_seed"))
            self.y_test = np.array(self._get_data(self.test_path, "is_seed"))
            typer.secho(f"{ok}Data loaded", color=typer.colors.GREEN)
        else:
            typer.secho("Data already populated", color=typer.colors.YELLOW)
        typer.secho(
            f"{ok}{len(self.text_train)} examples loaded in training set",
            color=typer.colors.BLUE,
        )
        typer.secho(
            f"{ok}{len(self.text_test)} examples loaded in test set",
            color=typer.colors.BLUE,
        )


class TextVectorizer(DataLoader):
    """
    Vectorize data

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import TextVectorizer
        text_loader = TextVectorizer("configs/model_cnn.yaml")
        text_loader.vectorize_text()

        # check examples
        text_loader.x_train
        ```
    """

    tokenizer = None
    vectorizer = None
    selector = None
    num_features = None
    x_train = None
    x_test = None
    max_length = None

    def __init__(self, config: Path):
        super().__init__(config)
        self.model_architecture = self.cfg["model"]["architecture"]

    def _fit_ngram_vectorizer(self):
        """Fit n-gram vectorizer

        !!! info
            Doc: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"""
        if self.vectorizer is None:
            kwargs = {
                k: self.cfg["tok2vec"][k]
                for k in [
                    "ngram_range",
                    "dtype",
                    "strip_accents",
                    "decode_error",
                    "analyzer",
                    "min_df",
                ]
            }
            self.vectorizer = TfidfVectorizer(**kwargs).fit(self.text_train)
        else:
            typer.secho(f"vectorizer already fitted", color=typer.colors.YELLOW)

    def _fit_ngram_feature_selector(self):
        """Fit feature selector

        !!! info
            Doc: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"""
        if self.selector is None:
            self.selector = SelectKBest(
                f_classif, k=min(self.cfg["tok2vec"]["top_k"], self.x_train.shape[1])
            ).fit(self.x_train, self.y_train)
        else:
            typer.secho(f"selector already fitted", color=typer.colors.YELLOW)

    def _fit_sequence_tokenizer(self):
        """Fit text tokenizer

        !!! info
            Doc: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer"""
        if self.tokenizer is None:
            # Create vocabulary with training texts.
            self.tokenizer = text.Tokenizer(num_words=self.cfg["tok2vec"]["top_k"])
            self.tokenizer.fit_on_texts(self.text_train)
        else:
            typer.secho(f"tokenizer already fitted", color=typer.colors.YELLOW)

    def _get_ngrams(self):
        """
        Return sparse matrices where 1 row corresponds to the tf-idf representation of a document with
        the number of columns corresponding to the cardinal of the (ngram) vocabulary.
        Note: Used in the MLP model only
        """
        if any(
            map(
                lambda x: x is None,
                [self.vectorizer, self.selector, self.x_train, self.x_test],
            )
        ):
            self.load_data()
            self._fit_ngram_vectorizer()

            # Learn vocabulary from training texts and vectorize training texts.
            self.x_train = self.vectorizer.transform(self.text_train)
            # Vectorize test texts.
            self.x_test = self.vectorizer.transform(self.text_test)

            # Find top 'k' vectorized features.
            self._fit_ngram_feature_selector()

            # Keep only top k features
            self.x_train = self.selector.transform(self.x_train).astype("float32")
            self.x_test = self.selector.transform(self.x_test).astype("float32")

        else:
            typer.secho(
                f"Data already ngramed. See self.x_train and self.x_test",
                color=typer.colors.YELLOW,
            )

    def _get_sequences(self):
        """
        Return the padded sequence of texts (train & test) with indices mapping to vocabulary (0 reserved for empty)
        and the related word-index mapping.
        Note: Used in the CNN model only
        """
        if any(
            map(
                lambda x: x is None,
                [
                    self.x_train,
                    self.x_test,
                    self.tokenizer,
                    self.num_features,
                    self.max_length,
                ],
            )
        ):
            self.load_data()
            self._fit_sequence_tokenizer()
            self.num_features = min(
                len(self.tokenizer.word_index) + 1, self.cfg["tok2vec"]["top_k"]
            )

            # Vectorize training and test texts.
            self.x_train = self.tokenizer.texts_to_sequences(self.text_train)
            self.x_test = self.tokenizer.texts_to_sequences(self.text_test)

            # Get max sequence length.
            self.max_length = len(max(self.x_train, key=len))
            # TODO: Very sensitive to outliers. Trimming above p90 might be more appropriate.
            if self.max_length > self.cfg["tok2vec"]["max_length"]:
                self.max_length = self.cfg["tok2vec"]["max_length"]

            # Fix sequence length to max value. Sequences shorter than the length are
            # padded in the beginning and sequences longer are truncated
            # at the beginning.
            self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.max_length)
            self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_length)
        else:
            typer.secho(
                "Sequences already populated. See self.x_train and self.x_test",
                color=typer.colors.YELLOW,
            )

    def vectorize_text(self):
        """Return vectorized texts (train and test)"""
        if self.model_architecture == "cnn":
            self._get_sequences()
        elif self.model_architecture == "mlp":
            self._get_ngrams()
        else:
            raise UnknownModel(UNKNOWN_MODEL_MSG)
        typer.secho(f"{ok}Text vectorized", color=typer.colors.GREEN)


class ModelBuilder(TextVectorizer):
    """
    Build model

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import ModelBuilder
        model_builder = ModelBuilder("configs/model_cnn.yaml")
        model_builder.build_model()

        # check model
        model_builder.model.summary()
        ```

    !!! info "Resources"
        [Google ML Guide on text-classification](https://developers.google.com/machine-learning/guides/text-classification/)
        [Keras text-classification](https://realpython.com/python-keras-text-classification/)
        [Understanding CNN](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/)
    """

    model = None

    def __init__(self, config: Path):
        super().__init__(config)

    def _build_mlp(self):  # -> models.Sequential
        """
        Instantiate a Multi layer perceptron Keras model
        """

        # Init model
        self.vectorize_text()  # we need to trigger it now to get x_train to determine the input shape
        self.input_shape = self.x_train.shape[1:]
        self.model = models.Sequential()
        self.model.add(
            Dropout(
                rate=self.cfg["model"]["dropout_rate"], input_shape=self.input_shape
            )
        )

        # Add mlp layers
        for _ in range(self.cfg["model"]["layers"] - 1):
            self.model.add(Dense(units=self.cfg["model"]["units"], activation="relu"))
            self.model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))

        # Binary classification + output ~ proba
        self.model.add(Dense(units=1, activation="sigmoid"))

    def _build_cnn(self, embedding_matrix: dict = None):  # -> models.Sequential
        """
        Instantiate a CNN model with <blocks> Convolution-Pooling pair layers.
        """

        self.model = models.Sequential()
        if self.cfg["model"]["use_pretrained_embedding"]:
            self.model.add(
                Embedding(
                    input_dim=self.num_features,
                    output_dim=self.cfg["model"]["embedding_dim"],
                    input_length=self.input_shape[0],
                    weights=[embedding_matrix],
                    trainable=self.cfg["model"]["is_embedding_trainable"],
                )
            )
        else:
            self.model.add(
                Embedding(
                    input_dim=self.num_features,
                    output_dim=self.cfg["model"]["embedding_dim"],
                    input_length=self.input_shape[0],
                )
            )
        for _ in range(self.cfg["model"]["blocks"] - 1):
            self.model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))
            self.model.add(
                Conv1D(
                    filters=self.cfg["model"]["filters"],
                    kernel_size=self.cfg["model"]["kernel_size"],
                    activation="relu",
                    bias_initializer="random_uniform",
                    padding="same",
                )
            )
            self.model.add(MaxPooling1D(pool_size=self.cfg["model"]["pool_size"]))

        self.model.add(
            Conv1D(
                filters=self.cfg["model"]["filters"] * 2,
                kernel_size=self.cfg["model"]["kernel_size"],
                activation="relu",
                bias_initializer="random_uniform",
                padding="same",
            )
        )
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))
        self.model.add(Dense(1, activation="sigmoid"))

    def build_model(self, embedding_matrix: dict = None):
        """
        Instantiate model based on config file
        """
        assert self.cfg["model"]["architecture"] in SupportedModels._member_names_
        if not self.model:
            if self.cfg["model"]["architecture"] == SupportedModels.mlp.value:
                self._build_mlp()
            elif self.cfg["model"]["architecture"] == SupportedModels.cnn.value:
                # TODO handle embedding matrix properly
                self._build_cnn(embedding_matrix)
            else:
                raise UnknownModel(UNKNOWN_MODEL_MSG)
            typer.secho(
                f"{ok}Model built (see self.model.summary() for details)",
                color=typer.colors.GREEN,
            )
        else:
            typer.secho("Model already built", color=typer.colors.YELLOW)


class ModelCompiler(ModelBuilder):
    """
    Compile model

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import ModelCompiler
        model_compiler = ModelCompiler("configs/model_cnn.yaml")
        model_compiler.compile_model()

        # check model, e.g. loss
        model_compiler.model.loss
        ```
    """

    optimizer = None

    def __init__(self, config: Path):
        super().__init__(config)

    def compile_model(self, embedding_matrix: dict = None):
        """Compile model. Use config file to instantiate training components."""
        self.build_model(embedding_matrix)
        self.optimizer = Adam(
            lr=float(self.cfg["training"]["optimizer"]["learning_rate"])
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.cfg["training"]["optimizer"]["loss"],
            metrics=self.cfg["training"]["optimizer"]["metrics"],
        )
        typer.secho(
            f"{ok}Model compiled (see self.model.summary() for details)",
            color=typer.colors.GREEN,
        )


class ModelFitter(ModelCompiler):
    """
    Fit model

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import ModelFitter
        model_fitter = ModelFitter("configs/model_cnn.yaml")
        model_fitter.fit_model()

        # check model, e.g. history
        model_fitter.model.history
        ```
    """

    callbacks = None

    def __init__(self, config: Path):
        super().__init__(config)

    def fit_model(self):
        """Fit model"""
        self.compile_model()
        self.callbacks = [
            EarlyStopping(
                monitor=self.cfg["training"]["callbacks"]["monitor"],
                patience=self.cfg["training"]["callbacks"]["patience"],
            )
        ]

        if not self.model.history:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.cfg["training"]["epochs"],
                callbacks=self.callbacks,
                validation_data=(self.x_test, self.y_test),
                verbose=self.cfg["logger"]["verbose"],
                batch_size=self.cfg["training"]["batch_size"],
            )
            typer.secho(f"{ok}Model trained", color=typer.colors.GREEN)
        else:
            # Alternative: clear session (keras.backend.clear_session()) and retrain
            typer.secho(f"Model already trained", color=typer.colors.YELLOW)

    # TODO save model & config file in folder


class Model(ModelFitter):
    """Main model class (data + model architecture + training)

    Arguments:
        config: config file path
        filepath: saving model directory

    **Usage:**
        ```python
        from techlandscape.model import Model
        model = Model("configs/model_cnn.yaml", "models/default_cnn/")
        model.fit_model()
        model.save()
        ```
    """

    def __init__(self, config: Path, filepath: Path):
        super().__init__(config)
        self.filepath = filepath

    def save(self):
        if not self.model.history:
            typer.secho(
                "No fitted model yet, cannot save. Run self.fit_model() and try again.",
                err=True,
                color=typer.colors.RED,
            )
        else:
            self.model.save(self.filepath)

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
from techlandscape.utils import get_config
from techlandscape.exception import UnknownModel, UNKNOWN_MODEL_MSG


# TODO
#   improve using self as of ModelBuilder
#   Check working properly (as of textVectorizer for cnn, MLP ok)


class DataLoader:
    def __init__(self, config: Path):
        self.cfg = get_config(config)
        self.train_path = Path(self.cfg["data"]["train"])
        self.test_path = Path(self.cfg["data"]["test"])
        self.text_train = None
        self.text_test = None
        self.y_train = None
        self.y_test = None

    @staticmethod
    def get_data(path: Path, var: str):
        return [
            json.loads(line)[var] for line in path.open("r").read().split("\n") if line
        ]

    def load_data(self):
        self.text_train = self.get_data(self.train_path, "text")
        self.text_test = self.get_data(self.test_path, "text")
        self.y_train = np.array(self.get_data(self.train_path, "is_seed"))
        self.y_test = np.array(self.get_data(self.test_path, "is_seed"))

        return self


class TextVectorizer(DataLoader):
    def __init__(self, config: Path):
        super().__init__(config)
        self.model_architecture = self.cfg["model"]["architecture"]
        self.tokenizer = None
        self.vectorizer = None
        self.selector = None
        self.num_features = None
        self.x_train = None
        self.x_test = None

    def fit_ngram_vectorizer(self):
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
        return self

    def fit_ngram_feature_selector(self):
        self.selector = SelectKBest(
            f_classif, k=min(self.cfg["tok2vec"]["top_k"], self.x_train.shape[1])
        ).fit(self.x_train, self.y_train)
        return self

    def fit_sequence_tokenizer(self):
        # Create vocabulary with training texts.
        tokenizer = text.Tokenizer(num_words=self.cfg["tok2vec"]["top_k"])
        self.tokenizer = tokenizer.fit_on_texts(self.text_train)
        return self

    def get_ngrams(self):
        """
        Return sparse matrices where 1 row corresponds to the tf-idf representation of a document with
        the number of columns corresponding to the card of the (ngram) vocabulary.
        Nb: Used in the MLP model only
        """
        self.load_data()
        self.fit_ngram_vectorizer()

        # Learn vocabulary from training texts and vectorize training texts.
        self.x_train = self.vectorizer.transform(self.text_train)
        # Vectorize test texts.
        self.x_test = self.vectorizer.transform(self.text_test)

        # Find top 'k' vectorized features.
        self.fit_ngram_feature_selector()

        # Keep only top k features
        self.x_train = self.selector.transform(self.x_train).astype("float32")
        self.x_test = self.selector.transform(self.x_test).astype("float32")

        return self

    def get_sequences(self):
        """
        Return the padded sequence of texts (train & test) with indices mapping to vocabulary (0 reserved for empty)
        and the related word-index mapping.
        """
        self.load_data()
        self.fit_sequence_tokenizer()
        self.num_features = min(
            len(self.tokenizer.word_index) + 1, self.cfg["tok2vec"]["top_k"]
        )

        # Vectorize training and test texts.
        self.x_train = self.tokenizer.texts_to_sequences(self.text_train)
        self.x_test = self.tokenizer.texts_to_sequences(self.text_test)

        # Get max sequence length.
        max_length = len(max(self.x_train, key=len))
        # TODO: Very sensitive to outliers. Trimming above p90 might be more appropriate.
        if max_length > self.cfg["tok2vec"]["max_length"]:
            max_length = self.cfg["tok2vec"]["max_length"]

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=max_length)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=max_length)
        return self

    def vectorize_text(self):
        """Return vectorized texts (train and test)"""
        if self.model_architecture == "cnn":
            self.get_sequences()
        elif self.model_architecture == "mlp":
            self.get_ngrams()
        else:
            UnknownModel(UNKNOWN_MODEL_MSG)
        return self


class ModelBuilder(TextVectorizer):
    """
        V0 based on https://developers.google.com/machine-learning/guides/text-classification/
        Other useful resources https://realpython.com/python-keras-text-classification/
        On CNN http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform
        -text-classification-with-word-embeddings/
    """

    def __init__(self, config: Path):
        super().__init__(config)
        self.x_train, self.x_test = self.vectorize_text()
        self.input_shape = self.x_train.shape[1:]

    def build_mlp(self) -> models.Sequential:
        """
        Return a Multi layer perceptron Keras model
        """

        # Init model
        model = models.Sequential()
        model.add(
            Dropout(
                rate=self.cfg["model"]["dropout_rate"], input_shape=self.input_shape
            )
        )

        # Add mlp layers
        for _ in range(self.cfg["model"]["layers"] - 1):
            model.add(Dense(units=self.cfg["model"]["units"], activation="relu"))
            model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))

        # Binary classification + output ~ proba
        model.add(Dense(units=1, activation="sigmoid"))
        return model

    def build_cnn(self, embedding_matrix: dict = None) -> models.Sequential:
        """
        Return a CNN model with <blocks> Convolution-Pooling pair layers.
        """
        # TODO input_shape and num_features should be inferred or declared
        #  input_shape: shape of input to the model.
        #  num_features: number of words (number of columns in embedding input).

        model = models.Sequential()
        if self.cfg["model"]["use_pretrained_embedding"]:
            model.add(
                Embedding(
                    input_dim=self.num_features,
                    output_dim=self.cfg["model"]["embedding_dim"],
                    input_length=self.input_shape[0],
                    weights=[embedding_matrix],
                    trainable=self.cfg["model"]["is_embedding_trainable"],
                )
            )
        else:
            model.add(
                Embedding(
                    input_dim=self.num_features,
                    output_dim=self.cfg["model"]["embedding_dim"],
                    input_length=self.input_shape[0],
                )
            )
        for _ in range(self.cfg["model"]["blocks"] - 1):
            model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))
            model.add(
                Conv1D(
                    filters=self.cfg["model"]["filters"],
                    kernel_size=self.cfg["model"]["kernel_size"],
                    activation="relu",
                    bias_initializer="random_uniform",
                    padding="same",
                )
            )
            model.add(MaxPooling1D(pool_size=self.cfg["model"]["pool_size"]))

        model.add(
            Conv1D(
                filters=self.cfg["model"]["filters"] * 2,
                kernel_size=self.cfg["model"]["kernel_size"],
                activation="relu",
                bias_initializer="random_uniform",
                padding="same",
            )
        )
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=self.cfg["model"]["dropout_rate"]))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def build_model(self, embedding_matrix: dict = None):
        assert self.cfg["model"]["architecture"] in ["mlp", "cnn"]
        if self.cfg["model"]["architecture"] == "mlp":
            model = self.build_mlp()
        elif self.cfg["model"]["architecture"] == "cnn":
            model = self.build_cnn(embedding_matrix)
        else:
            raise UnknownModel(UNKNOWN_MODEL_MSG)
        return model


class ModelCompiler(ModelBuilder):
    def __init__(self, config: Path):
        super().__init__(config)
        self.optimizer = Adam(lr=self.cfg["training"]["optimizer"]["learning_rate"])

    def compile_model(self, embedding_matrix: dict = None):
        model = self.build_model(embedding_matrix)
        model.compile(
            optimizer=self.optimizer,
            loss=self.cfg["training"]["optimizer"]["loss"],
            metrics=self.cfg["training"]["optimizer"]["metrics"],
        )
        return model


class ModelTrainer(ModelCompiler):
    def __init__(self, config: Path):
        super().__init__(config)
        self.callbacks = [
            EarlyStopping(
                monitor=self.cfg["training"]["callbacks"]["monitor"],
                patience=self.cfg["training"]["callbacks"]["patience"],
            )
        ]

    def train_model(self):
        # TODO find a way to keep track of history -> do it with self
        model = self.compile_model()
        model.fit(
            self.x_train,
            self.y_train,
            epochs=self.cfg["training"]["epochs"],
            callbacks=self.callbacks,
            validation_data=(self.x_test, self.y_test),
            verbose=self.cfg["logger"]["verbose"],
            batch_size=self.cfg["training"]["batch_size"],
        )
        return model

    # TODO save model & config file in folder

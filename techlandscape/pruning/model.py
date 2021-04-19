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


class DataLoader:
    def __init__(self, config: Path):
        self.cfg = get_config(config)
        self.model_architecture = self.cfg["model"]["architecture"]
        self.train_path = Path(self.cfg["data"]["train"])
        self.test_path = Path(self.cfg["data"]["test"])
        self.text_train = self._load_data(self.train_path, "text")
        self.text_test = self._load_data(self.test_path, "text")
        self.y_train = np.array(self._load_data(self.train_path, "is_seed"))
        self.y_test = np.array(self._load_data(self.test_path, "is_seed"))

    @staticmethod
    def _load_data(path: Path, var: str):
        return [
            json.loads(line)[var] for line in path.open("r").read().split("\n") if line
        ]

    def get_data(self):
        return self.text_train, self.text_test, self.y_train, self.y_test


class TextVectorizer(DataLoader):
    def __init__(self, config: Path):
        super().__init__(config)
        if self.model_architecture == "cnn":
            self.tokenizer = self.fit_sequence_tokenizer()
            self.num_features = self.get_num_features()

    def fit_ngram_vectorizer(self):
        kwargs = {
            k: self.cfg["tok2vec"][k]
            for k in [
                "ngram_range",
                "dtype",
                "strip_accents",
                "decode_error",
                "analyser",
                "min_df",
            ]
        }
        vectorizer = TfidfVectorizer(**kwargs)
        vectorizer.fit(self.text_train)
        return vectorizer

    def fit_ngram_feature_selector(self, x_train: np.array = None):
        if not x_train:  # just a hook to avoid regenerating x_train in already computed
            vectorizer = self.fit_ngram_vectorizer()
            x_train = vectorizer.transform(self.text_train)
        selector = SelectKBest(
            f_classif, k=min(self.cfg["tok2vec"]["top_k"], x_train.shape[1])
        )
        selector.fit(x_train, self.y_train)
        return selector

    def fit_sequence_tokenizer(self):
        # Create vocabulary with training texts.
        tokenizer = text.Tokenizer(num_words=self.cfg["tok2vec"]["top_k"])
        tokenizer.fit_on_texts(self.text_train)
        return tokenizer

    def get_num_features(self):
        return min(len(self.tokenizer.word_index) + 1, self.cfg["tok2vec"]["top_k"])

    def get_ngrams(self):
        """
        Return sparse matrices where 1 row corresponds to the tf-idf representation of a document with
        the number of columns corresponding to the card of the (ngram) vocabulary.
        Nb: Used in the MLP model only
        """

        vectorizer = self.fit_ngram_vectorizer()

        # Learn vocabulary from training texts and vectorize training texts.
        x_train = vectorizer.transform(self.text_train)
        # Vectorize test texts.
        x_test = vectorizer.transform(self.text_test)

        # Find top 'k' vectorized features.
        selector = self.fit_ngram_feature_selector(x_train=x_train)

        # Keep only top k features
        x_train = selector.transform(x_train).astype("float32")
        x_test = selector.transform(x_test).astype("float32")

        return x_train, x_test

    def get_sequences(self):
        """
        Return the padded sequence of texts (train & test) with indices mapping to vocabulary (0 reserved for empty)
        and the related word-index mapping.
        """

        tokenizer = self.fit_sequence_tokenizer()

        # Vectorize training and test texts.
        x_train = tokenizer.texts_to_sequences(self.text_train)
        x_test = tokenizer.texts_to_sequences(self.text_test)

        # Get max sequence length.
        max_length = len(max(x_train, key=len))
        # TODO: Very sensitive to outliers. Trimming above p90 might be more appropriate.
        if max_length > self.cfg["tok2vec"]["max_length"]:
            max_length = self.cfg["tok2vec"]["max_length"]

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        x_train = sequence.pad_sequences(x_train, maxlen=max_length)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
        return x_train, x_test

    def vectorize_text(self):
        """Return vectorized texts (train and test)"""
        if self.model_architecture == "cnn":
            x_train, x_test = self.get_sequences()
        elif self.model_architecture == "mlp":
            x_train, x_test = self.get_ngrams()
        else:
            UnknownModel(UNKNOWN_MODEL_MSG)
        return x_train, x_test


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
        # TODO see if input_shape could be inferred from config file params

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
        # TODO find a way to keep track of history
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

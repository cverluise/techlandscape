import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from techlandscape.utils import get_config, ok, not_ok, get_project_root
from techlandscape.exception import UnknownModel, UNKNOWN_MODEL_MSG
from techlandscape.enumerators import SupportedModels
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from smart_open import open
import typer
import hydra

app = typer.Typer()


class DataLoader:
    """
    Load data (train, test)

    Arguments:
        config: config file path

    **Usage:**
        ```python
        from techlandscape.model import DataLoader
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        data_loader = DataLoader(cfg)
        data_loader.load()

        # check examples
        data_loader.text_train
        ```
    """

    text_train = None
    text_test = None
    y_train = None
    y_test = None

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.train_path = get_project_root() / Path(
            self.cfg["data"]["train"]
        )  # Path(hydra.utils.get_original_cwd())
        self.test_path = get_project_root() / Path(
            self.cfg["data"]["test"]
        )  # Path(hydra.utils.get_original_cwd())

    @staticmethod
    def _get_data(path: Path, var: str):
        def file_reader(path: Path):
            with open(path, "r") as lines:  # that way we support gz files
                for line in lines:
                    try:
                        yield json.loads(line)
                    except json.decoder.JSONDecodeError:
                        pass

        return [line.get(var, "") for line in file_reader(path)]

    def load(self):
        """Load data. Expect a jsonl file where each row at least two fields: 'text' and 'is_seed'."""
        if any(
            map(
                lambda x: x is None,
                [self.text_train, self.text_test, self.y_train, self.y_test],
            )
        ):
            self.text_train = self._get_data(self.train_path, "text")
            self.text_test = self._get_data(self.test_path, "text")
            self.y_train = np.array(self._get_data(self.train_path, "is_seed")).astype(
                int
            )
            try:
                self.y_test = np.array(
                    list(filter(lambda x: x, self._get_data(self.test_path, "is_seed")))
                ).astype(int)
            except KeyError:
                typer.secho(
                    "No output variable in test. You can still vectorize the data.",
                    color=typer.colors.YELLOW,
                )
                pass
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
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        text_loader = TextVectorizer(cfg)
        text_loader.vectorize()

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
    checkpoint = None

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_architecture = self.cfg["model"]["architecture"]

    @staticmethod
    def _convert_sparse_matrix_to_sparse_tensor(x):
        coo = x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

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
            self.load()
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

            # to sparse tensor + reorder
            self.x_train = tf.sparse.reorder(
                self._convert_sparse_matrix_to_sparse_tensor(self.x_train)
            )
            self.x_test = tf.sparse.reorder(
                self._convert_sparse_matrix_to_sparse_tensor(self.x_test)
            )

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
            self.load()
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

    def _tokenize_from_pretrained(self):
        if not all([self.checkpoint, self.tokenizer, self.x_train, self.x_test]):
            self.load()
            self.checkpoint = self.cfg["model"]["checkpoint"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.x_train = dict(
                self.tokenizer(
                    self.text_train, padding=True, truncation=True, return_tensors="tf"
                )
            )
            self.x_test = dict(
                self.tokenizer(
                    self.text_test, padding=True, truncation=True, return_tensors="tf"
                )
            )
        else:
            typer.secho(
                "Sequences already tokenized. See self.x_train and self.x_test",
                color=typer.colors.YELLOW,
            )

    def vectorize(self):
        """Return vectorized texts (train and test)"""
        if self.model_architecture == SupportedModels.cnn.value:
            self._get_sequences()
        elif self.model_architecture == SupportedModels.mlp.value:
            self._get_ngrams()
        elif self.model_architecture == SupportedModels.transformers.value:
            self._tokenize_from_pretrained()
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
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        model_builder = ModelBuilder(cfg)
        model_builder.build()

        # check model
        model_builder.model.summary()
        ```

    !!! info "Resources"
        [Google ML Guide on text-classification](https://developers.google.com/machine-learning/guides/text-classification/)
        [Keras text-classification](https://realpython.com/python-keras-text-classification/)
        [Understanding CNN](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/)
    """

    model = None

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _build_mlp(self):  # -> models.Sequential
        """
        Instantiate a Multi layer perceptron Keras model
        """

        # Init model
        self.vectorize()  # we need to trigger it now to get x_train to determine the input shape
        self.input_shape = self.x_train.shape[1:]
        self.model = models.Sequential()
        # self.model.add(
        #     Dropout(
        #         rate=self.cfg["model"]["dropout_rate"], input_shape=self.input_shape
        #     )
        # )
        ## No dropout layer because of this *** issue: https://github.com/tensorflow/tensorflow/issues/25980

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
        self.vectorize()  # we need to trigger it now to get x_train to determine the input shape
        self.input_shape = self.x_train.shape[1:]

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

    def build(self, embedding_matrix: dict = None):
        """
        Instantiate model based on config file
        """
        assert self.model_architecture in SupportedModels._member_names_
        if not self.model:
            if self.model_architecture == SupportedModels.mlp.value:
                self._build_mlp()
            elif self.model_architecture == SupportedModels.cnn.value:
                # TODO handle embedding matrix properly
                self._build_cnn(embedding_matrix)
            elif self.model_architecture == SupportedModels.transformers.value:
                self.vectorize()
                self.model = TFAutoModelForSequenceClassification.from_pretrained(
                    self.checkpoint
                )
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
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        model_compiler = ModelCompiler(cfg)
        model_compiler.compile()

        # check model, e.g. loss
        model_compiler.model.loss
        ```
    """

    optimizer = None

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def compile(self, embedding_matrix: dict = None):
        """Compile model. Use config file to instantiate training components."""
        if self.model_architecture != SupportedModels.transformers.value:
            self.build(embedding_matrix)
            self.optimizer = Adam(
                lr=float(self.cfg["model"]["optimizer"]["learning_rate"])
            )
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.cfg["model"]["optimizer"]["loss"],
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
        elif self.model_architecture == SupportedModels.transformers.value:
            self.build(embedding_matrix)
            num_train_steps = (
                len(self.text_train) // self.cfg["training"]["batch_size"]
            ) * self.cfg["training"]["epochs"]
            lr_scheduler = PolynomialDecay(
                initial_learning_rate=self.cfg["model"]["optimizer"][
                    "init_learning_rate"
                ],
                end_learning_rate=self.cfg["model"]["optimizer"]["init_learning_rate"],
                decay_steps=num_train_steps,
            )
            self.optimizer = Adam(learning_rate=lr_scheduler)
            loss = eval(self.cfg["model"]["optimizer"]["loss"])(
                from_logits=self.cfg["model"]["optimizer"]["from_logits"]
            )
            self.model.compile(
                optimizer=self.optimizer, loss=loss, metrics=["accuracy"]
            )
        else:
            raise UnknownModel(UNKNOWN_MODEL_MSG)

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
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        model_fitter = ModelFitter(cfg)
        model_fitter.fit()

        # check model, e.g. history
        model_fitter.model.history
        ```
    """

    logdir = None
    filepath_best = None
    save_best_only = None
    callbacks = []

    def __init__(self, config: DictConfig):
        super().__init__(config)

    # @staticmethod
    # def _convert_sparse_matrix_to_sparse_tensor(X):
    #     coo = X.tocoo()
    #     indices = np.mat([coo.row, coo.col]).transpose()
    #     return tf.SparseTensor(indices, coo.data, coo.shape)

    def fit(self):
        """Fit model"""
        self.compile()
        if self.cfg["training"]["callbacks"]["early_stopping"]["active"]:
            self.callbacks += [
                EarlyStopping(
                    monitor=self.cfg["training"]["callbacks"]["early_stopping"][
                        "monitor"
                    ],
                    patience=self.cfg["training"]["callbacks"]["early_stopping"][
                        "patience"
                    ],
                    restore_best_weights=True,
                )
            ]
        if self.cfg["training"]["callbacks"]["save_best_only"]["active"]:
            self.save_best_only = True
            self.filepath_best = (
                get_project_root() / Path(self.cfg["out"]) / Path("model-best")
            )
            self.callbacks += [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.filepath_best,
                    monitor=self.cfg["training"]["callbacks"]["save_best_only"][
                        "monitor"
                    ],
                    save_best_only=True,
                    verbose=self.cfg["training"]["callbacks"]["save_best_only"][
                        "verbose"
                    ],
                )
            ]
        if self.cfg["logger"]["tensorboard"]["active"]:
            self.logdir = get_project_root() / Path(
                self.cfg["logger"]["tensorboard"]["logdir"]
            )
            self.callbacks += [tf.keras.callbacks.TensorBoard(self.logdir)]

        if not self.model.history:
            # if self.model_architecture == "mlp":
            #     self.x_train = tf.sparse.reorder(self._convert_sparse_matrix_to_sparse_tensor(self.x_train))
            #     self.x_test = tf.sparse.reorder(self._convert_sparse_matrix_to_sparse_tensor(self.x_test))

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


class Model(ModelFitter):
    """Main model class (data + model architecture + training)

    Arguments:
        config: config
        filepath: saving model directory

    **Usage:**
        ```python
        from techlandscape.model import Model
        from techlandscape.utils import get_config

        cfg = get_config("configs/model_cnn/default.yaml")
        cfg.update({"data": {"train": "your-train.jsonl", "test": "your-test.jsonl"}, "out": "your-save-dir"})

        model = Model(cfg)
        model.fit()
        model.save()
        ```
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.filepath = (
            self.filepath_best
            if self.filepath_best
            else get_project_root() / Path(self.cfg["out"]) / Path("model-last")
        )

    def save(self):
        if not self.model.history:
            typer.secho(
                "No fitted model yet, cannot save. Run self.fit_model() and try again.",
                err=True,
                color=typer.colors.RED,
            )
        else:
            if self.model_architecture == SupportedModels.transformers.value:
                self.model.save_pretrained(self.filepath)
            else:
                self.model.save(self.filepath)

    def save_config(self):
        Path(self.filepath / Path("config.yaml")).open("w").write(
            OmegaConf.to_yaml(self.cfg)
        )

    def save_meta(self):
        with Path(self.filepath / Path("meta.json")).open("w") as fout:
            fout.write(
                json.dumps(
                    {
                        "performance": self.model.evaluate(
                            self.x_test, self.y_test, return_dict=True, verbose=0
                        )
                    },
                    indent=2,
                )
            )


@hydra.main(config_path="../configs")
def train(cfg: DictConfig) -> None:
    """
    Train and save model
    """
    model = Model(config=cfg)
    model.fit()
    if not model.save_best_only:
        # no need to save the model itself if model_best=True, the checkpoint takes care of saving the best only
        model.save()
    model.save_meta()
    model.save_config()


if __name__ == "__main__":
    train()

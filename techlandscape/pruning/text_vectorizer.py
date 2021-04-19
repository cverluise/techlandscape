from keras.preprocessing import sequence, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from techlandscape.utils import get_config
from pathlib import Path
import numpy as np
import json


class TextVectorizer:
    def __init__(self, config: Path):
        self.cfg = get_config(config)
        self.model_architecture = self.cfg["model"]["architecture"]
        self.train_path = Path(self.cfg["data"]["train"])
        self.test_path = Path(self.cfg["data"]["test"])
        self.text_train = self.get_data(self.train_path, "text")
        self.text_test = self.get_data(self.test_path, "text")
        self.y_train = self.get_data(self.train_path, "is_seed")
        self.y_test = self.get_data(self.test_path, "is_seed")

    def get_data(self, path: Path, var: str):
        return [
            json.loads(line)[var] for line in path.open("r").read().split("\n") if line
        ]

    def fit_ngram_vectorizer(self):
        kwargs = {
            k: self.cfg.get(k)
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
            f_classif, k=min(self.cfg.get("top_k"), x_train.shape[1])
        )
        selector.fit(x_train, self.y_train)
        return selector

    def fit_sequence_tokenizer(self):
        # Create vocabulary with training texts.
        tokenizer = text.Tokenizer(num_words=self.cfg.get("top_k"))
        tokenizer.fit_on_texts(self.text_train)
        return tokenizer

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
        if max_length > self.cfg.get("max_length"):
            max_length = self.cfg.get("max_length")

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        x_train = sequence.pad_sequences(x_train, maxlen=max_length)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
        return x_train, x_test

    def fit_transform(self):
        """Return vectorized texts (train and test)"""
        if self.model_architecture == "cnn":
            x_train, x_test = self.get_sequences()
        else:
            x_train, x_test = self.get_ngrams()
        return x_train, x_test

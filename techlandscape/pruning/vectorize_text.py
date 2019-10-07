from keras.preprocessing import sequence, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)
# Limit on the number of features. We use the top 20K features.
TOP_K = 5000
# Whether text should be split into word or character n-grams. One of 'word', 'char'.
TOKEN_MODE = "word"
# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2
# Limit on the length of text sequences. Sequences longer than this will be truncated.
MAX_SEQUENCE_LENGTH = 250


# TODO: enrich with transformers in output


def get_ngram(texts_train, y_train, texts_test):
    """
    Return sparse matrices where 1 row corresponds to the tf-idf representation of a document with
    the number of columns corresponding to the card of the (ngram) vocabulary.
    :param texts_train: List[str], list of training set texts.
    :param y_train: arr, training set labels.
    :param texts_test: List[str], list of test set texts.
    :return: Vectorized training and test texts.
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        "ngram_range": NGRAM_RANGE,  # Use 1-grams + 2-grams.
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": TOKEN_MODE,  # Split text into word tokens.
        "min_df": MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(texts_train)
    # Vectorize test texts.
    x_test = vectorizer.transform(texts_test)

    # Find top 'k' vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, y_train)

    # Keep only top k features
    x_train = selector.transform(x_train).astype("float32")
    x_test = selector.transform(x_test).astype("float32")

    return x_train, x_test


def get_sequence(texts_train, texts_test):
    """
    Return the padded sequence of texts with indices mapping to vocabulary (0 reserved for empty).
    :param texts_train: List[str], list of training set texts.
    :param texts_test: List[str], list of test set texts.
    :return: Vectorized training and test texts + word-index mapping.
    """

    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(texts_train)

    # Vectorize training and test texts.
    x_train = tokenizer.texts_to_sequences(texts_train)
    x_test = tokenizer.texts_to_sequences(texts_test)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    # TODO: Very sensitive to outliers. Trimming above p90 might be more appropriate.
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)
    return x_train, x_test, tokenizer.word_index

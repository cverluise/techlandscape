# MODELS

A `Model` is made of data, a tok2vec, a model architecture and training specificities. This page documents the built-in models used for the pruning step and how they can be used/parametrized through human-readable config files.

!!! info
    Under the hood, we use keras. Components `model` and `training` can easily be extended to other keras parameters.

## data

The `data` component refers to the data used for training/testing the model.  

Name    | Description   | Type
---|---|---
`train` | Training data file path | `Path`
`test`  | Test data file path | `Path`

??? example
    ```yaml
    data:
      train: data/train.jsonl
      test: data/test.jsonl
    ```

!!! info "Data format"
    train/test are expected to be JSONL files where each row contains a field `"text"` (the abstract) and a field `"is_seed"` (the manual annotation) with values 1 if the candidate was classified in the seed, 0 otherwise.
    
    ```json
    # Example
    {"text": "fluctuat nec mergitur", "is_seed": 1}
    ```

## tok2vec

The `tok2vec` component refers to the text vectorization stage - that is the process going from raw text to data that can be ingested by the model.

=== "mlp"

    In the mlp case, we vectorize the text data using the `top_k` features of a [`TfIdfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) fitted on the training data. The `top_k` features are determined by the [`f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif).   

    Name| Description | Recommended values |Type
    ---|---|---|---
    `ngram_range`   | The lower and upper boundary of the range of n-values for different n-grams to be extracted. | `[1,2]`, `[1,3]`|`List[int]` 
    `dtype`         | Type of the matrix returned. | `float32` |`numeric`
    `strip_accents` | Remove accents and perform other character normalization during the preprocessing step. `"unicode"` is slightly slower but works on any character. | `"unicode"` | `{"ascii", "unicode"}`
    `decode_error`  | Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding. | `"replace"` | `{"strict", "ignore", "replace"}`
    `analyzer`      | Whether the feature should be made of word or character n-grams. | `"word"` | `{"word", "char", "char_wb"}`
    `min_df`        | When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.| `1`, `2`, `4` | `int` 
    `top_k`         | Select features according to the k f_classif highest scores. | `5000` | `int`
    
    ??? example
        ```yaml
        tok2vec:
          ngram_range: [1,2]
          dtype: float32
          strip_accents: unicode
          decode_error: replace
          analyzer: word
          min_df: 2
          top_k: 5000
        ```

=== "cnn"

    In the cnn case, we tokenize the text data using [keras.preprocessing.text.Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) and truncate/padd the sequence to `max_length`. only the `top_k` tokens (by frequency) are used.
    
    Name| Description | Recommended values| Type
    ---|---|---|---
    `max_length`| Max length of a tokenized text seqence. Above that figure, sequence is truncated, below, it is padded. | `250` | `int`
    `top_k`     | Only the most common num_words-1 words will be kept. | `5000` | `int`
    
    ??? example
        ```yaml
        max_length: 250
        top_k: 5000
        ```

## model

The model component defines the model's architecture, hyper-parameters and optimizer.

=== "mlp"

     Name| Description | Recommended values|Type
    ---|---|---|---
    `architecture`  | High level model architecture family name. | `"mlp"` |`"mlp"`
    `layers`        | Number of hidden layers. Nb: if 0, then logistic regression. | `0`, `1`, `2`, `4` |`int`
    `units`         | Number of units per hidden layer. | `16`, `32`, `64` |`int`
    `dropout_rate`  | Fraction of the input units to drop. | `0`, `0.2` |`float`  
    
    ??? example
        ```yaml
        model:
            architecture: mlp
            layers: 1
            units: 64
            dropout_rate: .2
        ```

=== "cnn"
    
    Name| Description | Recommended values |Type
    ---|---|---|---
    `architecture`  | High level model architecture family name. | `"cnn"` | `"cnn"`
    `blocks`        | Number of Convolution-Pooling pairs. | `1`, `2`, `4` | `int`
    `filters`       | Dimensionality of the output space (i.e. the number of output filters in the convolution). | `16`, `32`, `64`| `int`
    `dropout_rate`  | Percentage of input units to drop at Dropout Layer. | `0`, `0.2` | `float`
    `embedding_dim` | Dimension of the embedding vectors (reco between 50 and 200). | `50`, `100`, `200` | `int`
    `kernel_size`   | Length of the 1D convolution window. | `2`, `4`, `8` | `int`
    `pool_size`     | Factor by which to downscale input at MaxPooling layer. | `2`, `3`, `4` | `int`
    `use_pretrained_embedding`  | True if pre-trained. Else False. | `False` | `bool`
    `is_embedding_trainable`    | Used only `use_pretrained_embedding`. True if pre-trained embedding is trainable. Else False | `False` | `bool`

    ??? example
        ```yaml
        model:
          architecture: cnn
          blocks: 2
          filters: 64
          dropout_rate: .2
          embedding_dim: 100
          kernel_size: 5
          pool_size: 3
          use_pretrained_embedding: False
          is_embedding_trainable: False 
        ```

### Optimizer

The optimizer is nested in the model section. The same parameters are available for both architectures   

Name| Description | Recommended values | Type
---|---|---|---
`learning_rate` | Learning rate. | `1e-3` | `float`
`loss`          | Loss of the output layer. `"binary_crossentropy"` is warmly recommended since we are in a binary classification setting. See [keras doc](https://keras.io/api/losses/) for available losses | `"binary_crossentropy"` | `str`
`metrics`       | Metrics to record in `model.history` (also used in callbacks for early stopping). See [keras doc](https://keras.io/api/metrics/) for available metrics. | `["accuracy"]` | `List[str]`

??? example
    ```yaml
    model:
      ...
      optimizer:
        learning_rate: 1e-3
        loss: binary_crossentropy
        metrics: [ "accuracy" ]
    ```

## Training

The training component sets the training parameters.

Name| Description | Recommended values | Type
---|---|---|---
`epochs`    |Number of training epochs (full training dataset iteration). | `20`, `50`, `100` | `int`
`batch_size`|Size of the batch used for between 2 model updates. | `16`, `32`,  `64` | `int`

??? example
    ```yaml
    training:
      epochs: 100
      batch_size: 64
    ```

### Callbacks

Callbacks component is nested in training. It sets the parameters used for early-stopping. Seed [keras doc](https://keras.io/api/callbacks/early_stopping/) for more. 

Name| Description | Recommended values |Type
---|---|---|---
`monitor`   |Quantity to be monitored. | `"val_loss"` | `"val_loss"`
`patience`  |Number of epochs with no improvement after which training will be stopped. | `2`, `5`| `int`


??? example
    ```yaml
    training:
      ...
      callbacks:
        monitor: val_loss
        patience: 2
    ```

## Logger

The component logger sets the verbosity level of the model training.

Name| Description | Recommended values |Type
---|---|---|---
`verbose`|Verbosity level of the model training. 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. | `2` | `int`

??? example
    ```yaml
    logger:
    verbose: 2 
    ```

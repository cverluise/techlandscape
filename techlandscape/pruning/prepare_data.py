from sklearn.model_selection import train_test_split

from techlandscape.exception import SmallSeed


def test_ratio(seed_size, min_size=100, threshold_size=250):
    """
    Return the antiseed size such that the seed never represents less than 10% of the sample
    between <min_size> and <threshold_size> (linear) and 20% of the sample above
    <threshold_size>
    (constant). Seeds with less than <min_size> patents raise an error.
    :param seed_size: int
    :param min_size: int
    :param threshold_size: int
    :return:
    """
    if seed_size < min_size:
        raise SmallSeed("Danger Zone: your seed is too small. Don't cross!")
    elif min_size <= seed_size < threshold_size:
        test_ratio_ = (
            0.5 - (seed_size - min_size) / (threshold_size - min_size) * 0.3
        )
    else:
        test_ratio_ = 0.2
    return test_ratio_


def expansion_level_to_label(x):
    return 0 if x == "SEED" else 1


def get_train_test(classif_df, random_state=42):
    """

    :param classif_df:
    :return:
    """
    var_required = ["abstract", "expansion_level"]
    for v in var_required:
        assert v in classif_df.columns

    # prepare labels
    classif_df["label"] = classif_df["expansion_level"].apply(
        lambda x: expansion_level_to_label(x)
    )
    # shuffle
    classif_df.sample(frac=1, random_state=random_state)

    # train test set
    X = classif_df["abstract"].to_list()
    y = classif_df["label"].to_list()
    test_size = test_ratio(len(classif_df.query("expansion_level=='SEED'")))
    texts_train, texts_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return texts_train, texts_test, y_train, y_test

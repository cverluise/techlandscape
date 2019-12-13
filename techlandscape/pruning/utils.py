from techlandscape.exception import SmallSeed

cnn_params_grid = {
    "learning_rate": [1e-3],
    "epochs": [100],
    "batch_size": [64],
    "blocks": [1, 2, 3],
    "filters": [64, 128],
    "dropout_rate": [0, 0.2],
    "embedding_dim": [100],
    "kernel_size": [3, 5, 7],
    "pool_size": [3],
}

cnn_params_grid_sm = {
    "learning_rate": [1e-3],
    "epochs": [100],
    "batch_size": [64],
    "blocks": [1],
    "filters": [64, 128],
    "dropout_rate": [0],
    "embedding_dim": [100],
    "kernel_size": [3, 5],
    "pool_size": [3],
}

mlp_params_grid_sm = {
    "learning_rate": [1e-3],
    "epochs": [100],
    "batch_size": [64],
    "layers": [1, 2],  # when 1, no hidden layer -> standard logistic
    "units": [64],
    "dropout_rate": [0, 0.2],
}


def get_share_test(seed_size, min_size=100, threshold_size=250):
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
        share_test = (
            0.5 - (seed_size - min_size) / (threshold_size - min_size) * 0.3
        )
    else:
        share_test = 0.2
    return share_test

from techlandscape.exception import SmallSeed


def get_antiseed_size(seed_size, min_size=100, threshold_size=250):
    """
    Return the antiseed size such that the seed never represents less than 10% of the sample
    between <min_size> and <threshold_size> (linear) and 20% of the sample above <threshold_size>
    (constant). Seeds with less than <min_size> patents raise an error.
    :param seed_size: int
    :param min_size: int
    :param threshold_size: int
    :return:
    """
    if seed_size < min_size:
        raise SmallSeed("Danger Zone: your seed is too small. Don't cross!")
    elif min_size <= seed_size < threshold_size:
        share_seed = (
            0.1 + (seed_size - min_size) / (threshold_size - min_size) * 0.1
        )
        antiseed_size = int(seed_size / share_seed)
    else:
        share_seed = 0.2
        antiseed_size = int(seed_size / share_seed)
    return antiseed_size

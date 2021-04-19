SMALL_SEED_MSG = "Danger Zone: your seed is too small. Don't cross!"
UNKNOWN_MODEL_MSG = (
    "Unknown model architecture. Make sure that your config file is properly defined."
)


class SmallSeed(Exception):
    pass


class UnknownModel(Exception):
    pass

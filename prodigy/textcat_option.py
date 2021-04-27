import prodigy
from prodigy.components.loaders import JSONL


@prodigy.recipe(
    "textcat.option",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
)
def sentiment(dataset, file_path):
    """Annotate the sentiment of texts using different mood options."""
    stream = JSONL(file_path)  # load in the JSONL file

    return {
        "dataset": dataset,  # save annotations in this dataset
        "view_id": "choice",  # use the choice interface
        "stream": stream,
    }

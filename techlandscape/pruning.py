from pathlib import Path
from techlandscape.utils import get_config
from techlandscape.enumerators import SupportedModels, PrimaryKey
from techlandscape.model import TextVectorizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
from glob import glob
import typer

app = typer.Typer()


@app.command()
def get_score(
    path: str,
    model: str,
    dest: str,
    batch_size: int = 512,
    primary_key: PrimaryKey = "family_id",
) -> None:
    """
    Return the `model`'s predicted scores.

    Arguments:
        path: data file path (jsonl, wildcard enabled)
        model: model to be used fro predictions
        dest: detsination of the output
        batch_size: size of the batch (for prediction)
        primary_key: primary key

    **Usage:**
        ```shell
        techlandscape pruning get-score "expansion_additivemanufacturing_*.jsonl.gz" models/additivemanufacturing_cohM6_mlp/model-best/ additivemanufacturing_cohM6_mlp.jsonl
        ```
    """
    files = glob(path)
    out = pd.DataFrame()
    cfg = get_config(Path(model) / Path("config.yaml"))
    if cfg["model"]["architecture"] == SupportedModels.transformers.value:
        model_ = TFAutoModelForSequenceClassification.from_pretrained(model)
    else:
        model_ = tf.keras.models.load_model(model)

    for file in files:
        cfg["data"]["test"] = file
        text_vectorizer = TextVectorizer(cfg)
        text_vectorizer.vectorize()

        pred = model_.predict(text_vectorizer.x_test, batch_size=batch_size)
        if cfg["model"]["architecture"] == SupportedModels.transformers.value:
            score = tf.nn.softmax(pred["logits"])
            # pred is n*2, each line is [logits_0, logits_1]. We transform to proba (score) using softmax
            pred = score[:, 1].numpy()
            # we keep only the proba of class 1

        primary_keys = text_vectorizer._get_data(Path(file), primary_key.value)
        assert len(primary_keys) == len(pred.flatten())
        tmp = pd.DataFrame(
            zip(primary_keys, pred.flatten()), columns=[primary_key.value, "score"]
        )
        out = out.append(tmp)

    # save
    out.to_json(dest, orient="records", lines=True)


if __name__ == "__main__":
    app()

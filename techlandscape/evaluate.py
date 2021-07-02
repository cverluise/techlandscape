import json
import typer
import pandas as pd
import tensorflow as tf
from pathlib import Path
from techlandscape.utils import get_config
from techlandscape.enumerators import SupportedModels
from techlandscape.model import TextVectorizer
from glob import glob
from transformers import TFAutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

app = typer.Typer()


@app.command()
def models_performance(
    path: str, markdown: bool = True, destination: str = None, title: str = None
):
    """
    Summarize models performance and save to csv/ print to stdout

    Arguments:
        path: path of the meta.json (wildcard enabled)
        markdown: whether the output should be printed to stdout as md or saved to `destination`
        destination: destination file path (used if `--no-markdown`)
        title: title of the table (used if `--markdown`)

    **Usage:**
        ```shell
        techlandscape evaluate models-performance "models/additivemanufacturing_*_cnn/model-best/meta.json" --markdown --title "additivemanufacturing - cnn"
        ```
    """
    files = glob(path)
    get_name = lambda x: x.split("/")[1]

    for i, file in enumerate(files):

        tmp = pd.DataFrame.from_dict(json.loads(Path(file).open("r").read())).rename(
            columns={"performance": get_name(file)}
        )
        if i == 0:
            out = tmp.copy()
        else:
            out = out.merge(tmp, left_index=True, right_index=True)
    out = out.T
    out = out[sorted(out.columns)]
    if len(files) > 1:
        out = out.describe()

    if markdown:
        typer.echo(f"\n### {title}\n")
        typer.echo(f"{out.round(2).to_markdown()}")
    else:
        out.to_csv(destination)


@app.command()
def trf_performance(path: str,):
    """
    Harmonize the metrics reported in the transformers meta.json with those in other models' architectures meta.json.
    Specifically, add the precision and recall metrics (of class 1) and rename accuracy as binary accuracy.

    Arguments:
         path: model folder path

    **Usage:**
        ```shell
        techlandscape evaluate trf-performance "models/*transformers/model-last/"
        ```

    !!! note
        Needed because the models prediction is a tuple (2) with the logits of each class. This kind of prediction is
        not supported by keras precision and recall metrics.

    """
    models = glob(path)
    for model_ in models:
        cfg = get_config(Path(model_) / Path("config.yaml"))
        assert cfg["model"]["architecture"] == SupportedModels.transformers.value

        model = TFAutoModelForSequenceClassification.from_pretrained(model_)
        text_vectorizer = TextVectorizer(cfg)
        text_vectorizer.vectorize()
        pred = model.predict(text_vectorizer.x_test, batch_size=100)
        score = tf.nn.softmax(pred["logits"])
        pred = [0 if x < 0.5 else 1 for x in score[:, 1].numpy()]

        p, r, _, _ = precision_recall_fscore_support(text_vectorizer.y_test, pred)

        with open(Path(model_) / Path("meta.json"), "r") as fin:
            meta = json.loads(fin.read())
            meta.update(
                {
                    "performance": {
                        "binary_accuracy": meta["performance"]["accuracy"],
                        "precision": p[1],
                        "recall": r[1],
                    }
                }
            )
            with open(Path(model_) / Path("meta.json"), "w") as fout:
                fout.write(json.dumps(meta))

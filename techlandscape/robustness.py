import sys
import json
import typer
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List
from itertools import combinations, repeat
from techlandscape.utils import get_bq_client, get_config, ok, not_ok
from techlandscape.enumerators import (
    OverlapAnalysisKind,
    OverlapAnalysisAxis,
    SupportedModels,
)
from techlandscape.model import TextVectorizer
from glob import glob
from transformers import TFAutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

app = typer.Typer()


class OverlapAnalysis:
    tables = None
    table_combinations = None
    pairwise_overlap_analysis = None
    pairwise_overlap_ratios = None
    batch_overlap_ratios = None

    def __init__(
        self, technology: str, credentials: Path, robustness_dataset: str = "robustness"
    ):
        self.client = get_bq_client(credentials)
        self.technology = technology
        self.robustness_dataset = robustness_dataset

    @staticmethod
    def _get_uid(table: str) -> str:
        return table.split(".")[-1].split("_")[0]

    def get_tables(self) -> List[str]:
        tables = [
            table.full_table_id.replace(":", ".")
            for table in self.client.list_tables(self.robustness_dataset)
        ]
        tables = list(filter(lambda x: self.technology in x, tables))
        self.tables = tables
        return self.tables

    def get_table_combinations(self):
        if self.tables is None:
            self.get_tables()
        self.table_combinations = list(combinations(self.tables, 2))
        return self.table_combinations

    def pairwise_overlap_query(self) -> str:
        def _pairwise_overlap_query(table_combination) -> str:
            left_table, right_table = table_combination
            left_uid, right_uid = list(map(self._get_uid, (left_table, right_table)))
            query = f"""
                  SELECT 
                  "{'_'.join([left_uid, right_uid])}" as name,
                  COUNT(l.family_id) as nb,
                  FROM `{left_table}` as l,
                  `{right_table}`as r
                  WHERE l.family_id = r.family_id
                  """
            return query

        if self.table_combinations is None:
            self.get_table_combinations()
        return "\nUNION ALL\n".join(
            map(_pairwise_overlap_query, self.table_combinations)
        )

    def len_query(self) -> str:
        def _len_query(table: str) -> str:
            uid = self._get_uid(table)
            query = f"""
                SELECT 
                  "{uid}" as name,
                  COUNT(family_id) as nb
                FROM 
                `{table}`
                """
            return query

        if self.tables is None:
            self.get_tables()
        return "\nUNION ALL\n".join(map(_len_query, self.tables))

    def batch_overlap_query(self) -> str:
        def _batch_overlap_query(reference, table):
            ref_uid, uid = list(map(self._get_uid, [reference, table]))
            query = f"""
            INNER JOIN `{table}` AS {uid}
            ON {ref_uid}.family_id={uid}.family_id
            """
            return query

        if self.tables is None:
            self.get_tables()
        reference = self.tables[0]
        ref_uid = self._get_uid(reference)
        tables_ = self.tables.copy()
        tables_.pop(0)

        query_prefix = f"""SELECT COUNT({ref_uid}.family_id) as nb
        FROM `{reference}` AS {ref_uid}"""
        query = "\n".join(
            [query_prefix]
            + list(map(_batch_overlap_query, repeat(reference, len(tables_)), tables_))
        )
        return query

    def get_pairwise_overlap_analysis(self):
        overlap_df = self.client.query(self.pairwise_overlap_query()).to_dataframe()
        len_df = self.client.query(self.len_query()).to_dataframe()

        overlap_df["left_name"] = overlap_df["name"].apply(lambda x: x.split("_")[0])
        overlap_df["right_name"] = overlap_df["name"].apply(lambda x: x.split("_")[1])
        overlap_df = overlap_df.merge(
            len_df,
            how="left",
            left_on="left_name",
            right_on="name",
            suffixes=("", "_left"),
        )
        overlap_df = overlap_df.merge(
            len_df,
            how="left",
            left_on="right_name",
            right_on="name",
            suffixes=("_overlap", "_right"),
        )
        overlap_df = overlap_df.drop(["name_left", "name_right"], axis=1)
        overlap_df["share_left"] = overlap_df["nb_overlap"] / overlap_df["nb_left"]
        overlap_df["share_right"] = overlap_df["nb_overlap"] / overlap_df["nb_right"]
        self.pairwise_overlap_analysis = overlap_df
        return self.pairwise_overlap_analysis

    def get_pairwise_overlap_ratios(self):
        if self.pairwise_overlap_analysis is None:
            self.get_pairwise_overlap_analysis()
        self.pairwise_overlap_ratios = self.pairwise_overlap_analysis[
            "share_left"
        ].append(self.pairwise_overlap_analysis["share_right"])
        return self.pairwise_overlap_ratios

    def get_batch_overlap_analysis(self):
        nb_overlap = (
            self.client.query(self.batch_overlap_query()).to_dataframe()["nb"].values[0]
        )
        len_df = self.client.query(self.len_query()).to_dataframe()
        self.batch_overlap_ratios = (
            len_df["nb"].apply(lambda x: nb_overlap / x).rename("share")
        )
        return self.batch_overlap_ratios


@app.command()
def get_overlap_analysis(
    technology: str,
    kind: OverlapAnalysisKind,
    credentials: Path,
    summary: bool = False,
    destination: Path = None,
    robustness_dataset: str = "robustness",
):
    """
    Return the overlap analysis of `technology`

    Arguments:
        technology: name of the technology (as
        kind: kind of overlap analysis
        credentials: BQ credentials file path
        destination: results destination file path (if None, stdout)
        summary: whether the full analysis or its summary should be saved
        robustness_dataset: name of the BQ 'robustness' dataset

    **Usage:**
        ```shell
        techlandscape robustness get-overlap-analysis <technology> <your-credentials> --destination <overlap_analysis.csv>
        ```
    """

    overlap_analysis = OverlapAnalysis(technology, credentials, robustness_dataset)
    if kind == OverlapAnalysisKind.pairwise:
        if summary:
            overlap_analysis.get_pairwise_overlap_ratios()
            res, index, header = (
                overlap_analysis.pairwise_overlap_ratios.describe(),
                True,
                False,
            )
        else:
            overlap_analysis.get_pairwise_overlap_analysis()
            res, index, header = overlap_analysis.pairwise_overlap_analysis, False, True
    else:
        if summary:
            overlap_analysis.get_batch_overlap_analysis()
            res, index, header = (
                overlap_analysis.batch_overlap_ratios.describe(),
                True,
                False,
            )
        else:
            overlap_analysis.get_batch_overlap_analysis()
            res, index, header = overlap_analysis.batch_overlap_ratios, False, True
    destination = destination if destination else sys.stdout
    res.to_csv(destination, index=index, header=header)


@app.command()
def wrap_overlap_analysis(
    path: str,
    axis: OverlapAnalysisAxis,
    destination: str = None,
    markdown: bool = False,
):
    """
    Wrap overlap analysis based on csv output of  `get_overlap_analysis`

    Arguments:
        path: path of the files with results to be wrapped (wildcard enablec)
        axis: axis of the main analysis
        destination: saving file path (print to stdout in None)
        markdown: whether to return as md or csv table

    **Usage:**
        ```shell
        techlandscape robustness wrap-overlap-analysis "outs/expansion_*robustness*.csv" --markdown
        ```
    """
    files = glob(path)

    get_technology = lambda f: f.split("_")[1]
    get_config = lambda f: f.split("_")[2].replace(".csv", "")

    technologies = sorted(set([get_technology(f) for f in files]))
    configs = sorted(set([get_config(f) for f in files]))

    for e in eval(axis.value):
        files_ = [f for f in files if e in f]
        tmp = pd.DataFrame()
        for file in files_:
            name = (
                get_config(file)
                if axis == OverlapAnalysisAxis.technologies
                else get_technology(file)
            )
            tmp = tmp.append(pd.read_csv(file, names=["var", name]).set_index("var").T)
        tmp.index.name = (
            "technologies" if axis == OverlapAnalysisAxis.configs else "configs"
        )
        tmp = tmp.sort_index().round(2)
        out = destination if destination else sys.stdout
        if markdown:
            typer.echo(f"\n\n## {e}\n")
            tmp.to_markdown(out)
        else:
            tmp.to_csv(out)


@app.command()
def get_prediction_analysis(models: str, data: str, destination: Path = None):
    """
    Return a csv file with predicted scores on `data` for all models matching the `models` pattern.

    Arguments:
        models: model folder path (wildcard enabled)
        data: data file path
        destination: destination file path

    **Usage:**
        ```shell
        techlandscape robustness get-prediction-analysis "models/additivemanufacturing_*_cnn/model-best" data/expansion_additivemanufacturing_sample.jsonl --destination outs/
        # will be saved as classification_additivemanufacturing_robustness_cnn.csv
        ```
    """
    get_technology = lambda x: x.split("/")[-2].split("_")[0]
    get_architecture = lambda x: x.split("/")[-2].split("_")[-1]
    models = glob(models)
    for i, model_ in enumerate(models):
        technology = get_technology(model_)
        architecture = get_architecture(model_)
        cfg = get_config(Path(model_) / Path("config.yaml"))

        if cfg["model"]["architecture"] == SupportedModels.transformers.value:
            model = TFAutoModelForSequenceClassification.from_pretrained(model_)
        else:
            model = tf.keras.models.load_model(model_)
        cfg["data"]["test"] = data

        text_vectorizer = TextVectorizer(cfg)
        text_vectorizer.vectorize()

        pred = model.predict(text_vectorizer.x_test, batch_size=100)
        if cfg["model"]["architecture"] == SupportedModels.transformers.value:
            score = tf.nn.softmax(pred["logits"])
            # pred is n*2, each line is [logits_0, logits_1]. We transform to proba (score) using softmax
            pred = score[:, 1].numpy()
            # we keep only the proba of class 1

        if i == 0:
            out = pd.DataFrame(pred, columns=[model_])
        else:
            out = out.merge(
                pd.DataFrame(pred, columns=[model_]), left_index=True, right_index=True
            )
        filename = f"classification_{technology}_robustness_{architecture}.csv"
        out.to_csv(Path(destination) / Path(filename))
        typer.secho(f"{ok}{Path(destination) / Path(filename)} saved")


@app.command()
def wrap_prediction_analysis(
    path: str, group_by_technology: bool = False, markdown: bool = True
):
    """
    Wrap prediction analysis

    Arguments:
        path: prediction analysis file path (wildcard enabled)
        group_by_technology: whether to group by technology or not
        markdown: whether to output wrapped analysis as markdown or csv

    !!! attention "Usage warning"
        - when group_by_technology True, make sure to filter only files referring to the same technology
        - csv not supported yet

    **Usage:**
        ```shell
        techlandscape robustness wrap-prediction-analysis outs/classification_additivemanufacturing_robustness_cnn.csv
        ```
    """
    get_technology = lambda x: x.split("/")[1].split("_")[1]
    get_architecture = lambda x: x.split("/")[1].split("_")[-1].split(".")[0]
    files = glob(path)
    files = sorted(files)
    dispersion_out, consensus_out = None, None

    for i, file in enumerate(files):
        technology = get_technology(file)
        architecture = get_architecture(file)

        tmp = pd.read_csv(file, index_col=0)
        dispersion = tmp.std(axis=1).describe().rename("std_score").copy()

        for col in tmp.columns:
            tmp[col] = tmp[col].apply(lambda x: 1 if x > 0.5 else 0)
        tmp["vote"] = tmp.sum(1)
        tmp = (
            tmp.groupby("vote")
            .count()
            .max(1)
            .to_frame()
            .reset_index()
            .prod(1)
            .rename("nb_positives")
            .to_frame()
        )
        tmp["share_positives"] = tmp["nb_positives"] / tmp["nb_positives"].sum()
        tmp = tmp[::-1]
        tmp["cumshare_positives"] = tmp["share_positives"].cumsum()
        tmp.index.name = "nb_models"
        consensus = tmp.copy()

        if group_by_technology:
            dispersion = dispersion.rename(architecture).to_frame()
            consensus = consensus.rename(columns={"cumshare_positives": architecture})[
                architecture
            ].to_frame()

            dispersion_out = (
                dispersion_out.merge(dispersion, right_index=True, left_index=True)
                if dispersion_out is not None
                else dispersion
            )
            consensus_out = (
                consensus_out.merge(
                    consensus, right_index=True, how="outer", left_index=True
                )
                if consensus_out is not None
                else consensus
            )

        else:
            if markdown:
                typer.echo(f"\n## {technology} - {architecture}\n")
                typer.echo("### Score dispersion\n")
                typer.echo(dispersion.round(3).to_markdown() + "\n")
                typer.echo("### Models consensus\n")
                typer.echo(consensus.round(3).to_markdown())
            else:
                typer.secho(f"{not_ok}csv not supported yet", err=True)

    if group_by_technology:
        if markdown:
            typer.echo(f"\n## {technology}\n")
            typer.echo("### Score dispersion\n")
            typer.echo(dispersion_out.round(3).to_markdown() + "\n")
            typer.echo("### Models consensus\n")
            typer.echo(
                consensus_out.sort_values("nb_models", ascending=False)
                .round(3)
                .to_markdown()
            )
        else:
            typer.secho(f"{not_ok}csv not supported yet", err=True)


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
        techlandscape robustness models-performance "models/additivemanufacturing_*_cnn/model-best/meta.json" --markdown --title "additivemanufacturing - cnn"
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
def get_trf_performance(path: str,):
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

        with open(Path(model_) / Path("config.yaml"), "r+") as fin:
            meta = json.loads(fin.read())
            meta.update(
                {"accuracy": meta.get("binary_accuracy"), "precision": p, "recall": r}
            )
            fin.write(json.dumps(meta))


if __name__ == "__main__":
    app()

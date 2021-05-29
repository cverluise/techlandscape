import sys
from pathlib import Path
from typing import List
from itertools import combinations, repeat
import typer
from techlandscape.utils import get_bq_client
from techlandscape.enumerators import OverlapAnalysisKind

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
            res, index = overlap_analysis.pairwise_overlap_ratios.describe(), True
        else:
            overlap_analysis.get_pairwise_overlap_analysis()
            res, index = overlap_analysis.pairwise_overlap_analysis, False
    else:
        if summary:
            overlap_analysis.get_batch_overlap_analysis()
            res, index = overlap_analysis.batch_overlap_ratios.describe(), True
        else:
            overlap_analysis.get_batch_overlap_analysis()
            res, index = overlap_analysis.batch_overlap_ratios, False
    destination = destination if destination else sys.stdout
    res.to_csv(destination, index=index)


if __name__ == "__main__":
    app()

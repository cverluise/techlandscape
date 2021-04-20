import yaml
import typer
from pathlib import Path
from techlandscape.utils import flatten, get_bq_job_done
from typing import List

app = typer.Typer()


class QueryCandidates:
    """Class """

    def __init__(self, config_file: Path):
        self.config_file = Path(config_file)
        self.config = self.get_config()
        self.patents = flatten([v.split(",") for _, v in self.config["patent"].items()])
        self.cpcs = flatten([v.split(",") for _, v in self.config["cpc"].items()])
        self.keywords = flatten(
            [v.split(",") for _, v in self.config["keyword"].items()]
        )

    def get_config(self) -> dict:
        """Load config file"""
        with self.config_file.open("r") as config:
            config = yaml.load(config, Loader=yaml.FullLoader)
        return config

    def get_query_patents(self, patents: List[str]) -> str:
        """Return candidates query based on patent similarity"""
        patents = patents if patents else self.patents
        query = f"""
        SELECT
          similar.publication_number,
          title,
          abstract,
          "patent" AS match
        FROM
          `patents-public-data.google_patents_research.publications` AS gpr,
          UNNEST(similar) AS similar
        WHERE
          gpr.publication_number IN ({",".join(map(lambda x: '"' + x + '"', patents))})
          """
        return query

    def get_query_keywords(self, keywords: List[str]) -> str:
        """Return candidates query based on keywords"""
        keywords = keywords if keywords else self.keywords
        query = f"""
            SELECT
              publication_number,
              title,
              abstract,
              "keyword" AS match
            FROM
              `patents-public-data.google_patents_research.publications` AS gpr
            WHERE {" OR ".join(map(lambda x: 'LOWER(gpr.abstract) LIKE "%' + x + '%"', keywords))}
        """
        return query

    def get_query_cpcs(self, cpcs: List[str]) -> str:
        """Return candidates query based on cpcs"""
        cpcs = cpcs if cpcs else self.cpcs
        query = f"""
            SELECT
              publication_number,
              title,
              abstract,
              "cpc" AS match
            FROM
              `patents-public-data.google_patents_research.publications` AS gpr,
              UNNEST(cpc) as cpc
            WHERE {" OR ".join(map(lambda x: 'cpc.code LIKE "' + x + '%"', cpcs))}
        """
        return query

    def get_query(self) -> str:
        """Return candidates query (all)"""
        query_patents = self.get_query_patents(self.patents)
        query_cpcs = self.get_query_cpcs(self.cpcs)
        query_keywords = self.get_query_keywords(self.keywords)

        query = f"""
        WITH tmp AS (
        {"UNION ALL".join([query_patents, query_cpcs, query_keywords])}
        )
        SELECT
          p.family_id,
          STRING_AGG(tmp.publication_number) AS publication_number,
          CONCAT(ANY_VALUE(tmp.title), "\\n\\n", ANY_VALUE(tmp.abstract)) AS text,
          STRING_AGG(DISTINCT(tmp.match)) AS match,
          ARRAY_LENGTH(SPLIT(STRING_AGG(tmp.match))) AS match_number,
        FROM
          tmp
        LEFT JOIN
          `patents-public-data.patents.publications` AS p
        ON
          tmp.publication_number=p.publication_number
        GROUP BY
          family_id
        """
        return query


@app.command()
def get_candidates(config_file: Path, destination_table: str, credentials: Path):
    """
    Return seed candidates based on `config_file`. Candidate table is saved to `destination_table`
    """
    query = QueryCandidates(config_file).get_query()
    get_bq_job_done(query, destination_table, credentials)


if __name__ == "__main__":
    app()

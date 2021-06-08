import typer
from pathlib import Path
from typing import List
from techlandscape.decorators import monitor
from techlandscape.utils import (
    get_country_string_bq,
    get_bq_client,
    get_project_id,
    get_bq_job_done,
)
from techlandscape.enumerators import PrimaryKey

app = typer.Typer()


@app.command()
def get_training_data(
        primary_key: PrimaryKey,
        seed_table: str,
        expansion_table: str,
        af_antiseed_size: int,
        destination_table: str,
        credentials: Path,
        verbose: bool = False,
) -> None:
    """
    Return training data

    Arguments:
        primary_key: table primary key
        seed_table: seed table (project.dataset.table)
        expansion_table: expansion table used for drawing the antiseed a la AF (project.dataset.table)
        af_antiseed_size: size of the antiseed a la AF
        destination_table: destination table
        credentials: BQ credentials file path
        verbose: verbosity

    **Usage:**
        ```shell
        techlandscape io get-training-data family_id <seed-table> <expansion-table> <af-seed-size> <destination-table> credentials_bq.json
        ```
    """
    project_id = get_project_id(primary_key, credentials)
    query = f"""
    WITH
      manual AS (
      SELECT
        {primary_key.value},
        expansion_level,
        CAST(expansion_level="SEED" AS INT64) AS is_seed,
        STRUCT(CAST(expansion_level="SEED" AS INT64) AS SEED,
          CAST(expansion_level LIKE "ANTISEED%" AS INT64) AS NOT_SEED) AS cats,
        text
      FROM
        `{seed_table}`
      WHERE
        expansion_level LIKE "%SEED%"),
      random AS (
    SELECT
      r.{primary_key.value} AS {primary_key.value},
      "ANTISEED-AF" AS expansion_level,
      CAST(0 AS INT64) AS is_seed,
      STRUCT(0 AS SEED,
        1 AS NOT_SEED) AS cats,
      abstract AS text
    FROM
      `{project_id}.patents.publications` AS r #country_prefix
    LEFT OUTER JOIN
      `{expansion_table}` AS tmp
    ON
      r.{primary_key.value} = tmp.{primary_key.value}
    WHERE
      abstract IS NOT NULL
      AND abstract != ""
    ORDER BY
      RAND()
    LIMIT
      {af_antiseed_size}
      )
    SELECT * FROM manual
    UNION ALL
    SELECT * FROM  random 
    """
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


@monitor
@app.command()
def get_expansion(primary_key: PrimaryKey,
                  table_ref: str,
                  destination_table: str,
                  credentials: Path,
                  sample_size: int = None,
                  verbose: bool = False
                  ) -> None:
    """
    Return (a sample of) the expansion table

    Arguments:
        primary_key: table primary key
        table_ref: expansion table
        destination_table: destination table
        credentials: credentials file path
        sample_size: size of the sample (if None, then we extract all)
        verbose: verbosity

    **Usage:**
        ```
        techlandscape io get-expansion family_id <table-ref> <destination-table> credentials_bq.json --sample-size 10000
        ```
    """
    project_id = get_project_id(primary_key, credentials)
    sample_clause = f"LIMIT {sample_size}" if sample_size else ""

    query = f"""
    WITH
      expansion AS (
      SELECT
        *
      FROM
        `{table_ref}` AS expansion  # patentcity.techdiffusion.expansion_additivemanufacturing
      WHERE
        expansion_level NOT LIKE "%SEED%" )
    SELECT
      expansion.*,
      p.abstract AS text
    FROM
      expansion
    LEFT JOIN
      `{project_id}.patents.publications` AS p
    ON
      expansion.{primary_key.value} = p.{primary_key.value}
    WHERE
      p.abstract IS NOT NULL
      AND p.abstract != ""
      AND p.abstract != "\\n"
      AND p.abstract != "\\n\\n"
    ORDER BY RAND()  
    {sample_clause}
    """
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


if __name__ == "__main__":
    app()

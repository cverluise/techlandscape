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
@app.command(deprecated=True)
def get_expansion_table(
    flavor: str, table_ref: str, countries: List[str], credentials: Path
):
    """
    Return the expansion(publication_number, expansion_level, abstract)
    `flavor` in ['*expansion', '*seed']
    """
    # TODO refacto
    #  - remove seed option
    #  - save to BQ stage and then extract using standard bq extract | gsutil cp workflow
    assert flavor in ["*expansion", "*seed"]

    expansion_level_clause = "" if flavor == "*seed" else "NOT"
    country_clause = (
        ""
        if flavor == "*seed"
        else (
            f"AND r.country in ({get_country_string_bq(countries)})"
            if countries
            else ""
        )
    )

    query = f"""
    SELECT
      tmp.publication_number,
      tmp.expansion_level,
      abstract
    FROM
      `patents-public-data.google_patents_research.publications` as r,
      {table_ref} as tmp
    WHERE
      r.publication_number=tmp.publication_number
      {country_clause}
      AND abstract is not NULL
      AND abstract!=''
      AND expansion_level {expansion_level_clause} LIKE "%SEED%"
    GROUP BY
      publication_number, expansion_level, abstract  
    """

    client = get_bq_client(credentials)
    return client.query(query).to_dataframe()


if __name__ == "__main__":
    app()

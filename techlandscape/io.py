import typer
from pathlib import Path
from techlandscape.decorators import monitor
from techlandscape.utils import get_project_id, get_bq_job_done
from techlandscape.enumerators import PrimaryKey, JoinHow

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
def get_expansion(
    primary_key: PrimaryKey,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    sample_size: int = None,
    verbose: bool = False,
) -> None:
    """
    Return (a sample of) the expansion table

    Arguments:
        primary_key: table primary key
        table_ref: expansion table (project.dataset.table)
        destination_table: destination table (project.dataset.table)
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


@app.command()
def join(
    left_table: str,
    right_table: str,
    on: str,
    destination_table: str,
    credentials: Path,
    how: JoinHow = "INNER",
    right_vars: str = None,
    abstract2text: bool = False,
    verbose: bool = False,
) -> None:
    """
    Return the join of `left_table` and `right_table`.

    Arguments:
        left_table: left table, rel to the join (project.dataset.table)
        right_table: right table, rel to the join (project.dataset.table)
        on: joining key
        destination_table: BQ destination table (project.dataset.table)
        credentials: credentials file path
        how: join method
        right_vars: comma separated var to keep from the right table
        abstract2text: whether to rename abstract in `right_table` as text (supported only when right_vars null)
        verbose: bool = False

    **Usage:**
        ```shell
        techlandscape io join <left-table> <right-table> <on> <destination-table>
        ```
    """
    right_vars_clause = (
        ", ".join([f"right_table.{v}" for v in right_vars.split(",")])
        if right_vars
        else (
            f"right_table.* EXCEPT({on})"
            if not abstract2text
            else f"right_table.* EXCEPT({on}, abstract), right_table.abstract as text"
        )
    )

    query = f"""
    SELECT
      left_table.*,
      {right_vars_clause}
    FROM
      `{left_table}` AS left_table  # patentcity.techdiffusion.expansion_additivemanufacturing
    {how.value} JOIN
      `{right_table}` AS right_table  # patentcity.patents.family_abstract
    ON
      left_table.{on} = right_table.{on}"""
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


if __name__ == "__main__":
    app()

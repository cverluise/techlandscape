import typer
from pathlib import Path
from typing import List
from techlandscape.query import (
    get_country_clause,
    get_pc_like_clause,
    get_country_prefix,
    get_project_id,
)
from techlandscape.utils import get_bq_client, get_bq_job_done

# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning

app = typer.Typer()


def get_af_antiseed_query(
    table_ref: str,
    size: int,
    src_project_id: str,
    key: str = "publication_number",
    countries: List[str] = None,
) -> str:
    """
    Return the anti-seed a la AF
    """
    assert key in ["publication_number", "family_id"]

    country_prefix = get_country_prefix(key)
    country_clause = get_country_clause(countries)
    query = f"""SELECT
      DISTINCT(r.{key}) AS {key},
      "ANTISEED-AF" AS expansion_level
    FROM
      `{src_project_id}.google_patents_research.publications` AS r {country_prefix}
    LEFT OUTER JOIN
      {table_ref} AS tmp
    ON
      r.{key} = tmp.{key}
    WHERE
      r.abstract is not NULL
      AND r.abstract!=''
      {country_clause}
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    return query


def get_aug_antiseed_query(
    table_ref: str,
    size: int,
    flavor: str,
    pc_list: List[str],
    src_project_id: str,
    key="publication_number",
    countries: List[str] = None,
) -> str:
    """
    Return the augmented anti-seed
    """
    assert flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    pc_like_clause_ = get_pc_like_clause(flavor, pc_list, sub_group=True)
    country_prefix = get_country_prefix(key)
    country_clause = get_country_clause(countries)
    query = f"""
    SELECT
      DISTINCT(r.{key}) AS {key},
      "ANTISEED-AUG" AS expansion_level
    FROM
      `{src_project_id}.google_patents_research.publications` AS r,
      UNNEST({flavor}) AS {flavor} {country_prefix}
    LEFT OUTER JOIN
      {table_ref} AS tmp
    ON
      r.{key} = tmp.{key}
    WHERE
      {pc_like_clause_}
      {country_clause}
      AND r.abstract is not NULL
      AND r.abstract!=''
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    return query


# @monitor
@app.command()
def get_antiseed(
    table_ref: str,
    destination_table: str,
    size: int,
    flavor: str,
    pc_list: List[str],
    credentials: Path,
    key: str = "publication_number",
    countries: List[str] = None,
):
    """
    Draw antiseed (a la AF & augmented)
    """
    client = get_bq_client(credentials)
    # job_config = bigquery.QueryJobConfig(destination=destination)

    src_project_id = get_project_id(key, client)
    af_antiseed_query = get_af_antiseed_query(
        table_ref, size, src_project_id, key=key, countries=countries
    )
    aug_antiseed_query = get_aug_antiseed_query(
        table_ref,
        size,
        flavor,
        pc_list,
        src_project_id,
        key=key,
        countries=countries,
    )
    get_bq_job_done(af_antiseed_query, destination_table, credentials)
    get_bq_job_done(
        aug_antiseed_query, destination_table, credentials, "WRITE_APPEND"
    )


if __name__ == "__main__":
    app()

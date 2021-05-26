import typer
from pathlib import Path
from typing import List
from techlandscape.utils import (
    get_country_clause,
    get_pc_like_clause,
    get_country_prefix,
    get_project_id,
    get_bq_job_done,
)
from techlandscape.enumerators import TechClass, PrimaryKey

# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning

app = typer.Typer()


def get_af_antiseed_query(
    table_ref: str,
    size: int,
    src_project_id: str,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    countries: List[str] = None,
) -> str:
    """
    Return the anti-seed a la AF
    """

    country_prefix = get_country_prefix(primary_key)
    country_clause = get_country_clause(countries)
    query = f"""SELECT
      DISTINCT(r.{primary_key.value}) AS {primary_key.value},
      "ANTISEED-AF" AS expansion_level
    FROM
      `{src_project_id}.google_patents_research.publications` AS r {country_prefix}
    LEFT OUTER JOIN
      {table_ref} AS tmp
    ON
      r.{primary_key.value} = tmp.{primary_key.value}
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
    tech_class: TechClass,
    pc_list: List[str],
    src_project_id: str,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    countries: List[str] = None,
) -> str:
    """
    Return the augmented anti-seed
    """

    pc_like_clause_ = get_pc_like_clause(tech_class, pc_list, sub_group=True)
    country_prefix = get_country_prefix(primary_key)
    country_clause = get_country_clause(countries)
    query = f"""
    SELECT
      DISTINCT(r.{primary_key.value}) AS {primary_key.value},
      "ANTISEED-AUG" AS expansion_level
    FROM
      `{src_project_id}.google_patents_research.publications` AS r,
      UNNEST({tech_class.value}) AS {tech_class.value} {country_prefix}
    LEFT OUTER JOIN
      {table_ref} AS tmp
    ON
      r.{primary_key.value} = tmp.{primary_key.value}
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
    tech_class: TechClass,
    pc_list: List[str],
    credentials: Path,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    countries: List[str] = None,
):
    """
    Draw antiseed (a la AF & augmented)
    """

    src_project_id = get_project_id(primary_key, credentials)
    af_antiseed_query = get_af_antiseed_query(
        table_ref, size, src_project_id, primary_key=primary_key, countries=countries
    )
    aug_antiseed_query = get_aug_antiseed_query(
        table_ref,
        size,
        tech_class,
        pc_list,
        src_project_id,
        primary_key=primary_key,
        countries=countries,
    )
    get_bq_job_done(af_antiseed_query, destination_table, credentials)
    get_bq_job_done(aug_antiseed_query, destination_table, credentials, "WRITE_APPEND")


if __name__ == "__main__":
    app()

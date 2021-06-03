import typer
from pathlib import Path
from techlandscape.utils import (
    get_pc_like_clause,
    get_country_prefix,
    get_project_id,
    get_bq_job_done,
)
from techlandscape.enumerators import TechClass, PrimaryKey


app = typer.Typer()


@app.command(deprecated=True)
def get_aug_antiseed(
    primary_key: PrimaryKey,
    tech_class: TechClass,
    pc_list: str,
    size: int,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    write_disposition: str = "WRITE_APPEND",
    verbose: bool = False,
) -> None:
    """
    !!! warning "Deprecated"
        The augmented antiseed is not used any more. We have "close" negative examples from human annotations.

    Return the augmented anti-seed

    Arguments:
        primary_key: table primary key
        tech_class: technological class considered
        pc_list: list of technological classes to draw from, comma-separated (e.g. A,B,C)
        size: size of the antiseed
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: credentials file path
        write_disposition: BQ write disposition
        verbose: verbosity

    **Usage:**
        ```shell
        techlandscape antiseed get-af-antiseed family_id cpc Y12,Y10 300 <expansion-table> <destination-table> credentials_bq.json
        ```
    """
    project_id = get_project_id(primary_key, credentials)
    pc_list = pc_list.split(",")

    pc_like_clause_ = get_pc_like_clause(tech_class, pc_list, sub_group=True)
    country_prefix = get_country_prefix(primary_key)
    query = f"""
    SELECT
      DISTINCT(r.{primary_key.value}) AS {primary_key.value},
      "ANTISEED-AUG" AS expansion_level
    FROM
      `{project_id}.patents.publications` AS r,
      UNNEST({tech_class.value}) AS {tech_class.value} {country_prefix}
    LEFT OUTER JOIN
      {table_ref} AS tmp
    ON
      r.{primary_key.value} = tmp.{primary_key.value}
    WHERE
      {pc_like_clause_}
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    get_bq_job_done(query, destination_table, credentials, write_disposition, verbose)


if __name__ == "__main__":
    app()

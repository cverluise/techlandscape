import typer
from pathlib import Path
from typing import List
from google.cloud import bigquery
from techlandscape.decorators import monitor, timer
from techlandscape.utils import get_country_string_bq, get_bq_client, ok

app = typer.Typer()


@monitor
@app.command()
def load_file_to_bq(
    seed_file: Path,
    destination: str,
    credentials: Path,
    write_disposition: str = "WRITE_TRUNCATE",
    source_format: str = "NEWLINE_DELIMITED_JSON",
    autodetect: bool = True,
    **kwargs,
):
    """
    Load file to BigQuery. Mainly a bigquery.client.load_table_from_file wrapper.
    """
    seed_file = Path(seed_file)
    client = get_bq_client(credentials)
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        source_format=source_format,
        autodetect=autodetect,
        **kwargs,
    )
    client.load_table_from_file(
        seed_file.open("rb"), destination=destination, job_config=job_config
    ).result()
    typer.secho(f"{ok}File saved to {destination}", fg=typer.colors.GREEN)


@timer
@monitor
@app.command()
def get_expansion_table(
    flavor: str, table_ref: str, countries: List[str], credentials: Path
):
    """
    Return the expansion(publication_number, expansion_level, abstract)
    `flavor` in ['*expansion', '*seed']
    """
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
    # TODO dump to stdout?
    return client.query(query).to_dataframe()


if __name__ == "__main__":
    app()

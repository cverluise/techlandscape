import typer
from pathlib import Path
from techlandscape.utils import get_bq_job_done

app = typer.Typer()


@app.command()
def get_patents_family(
    destination_table: str, credentials: Path, verbose: bool = False
):
    """Emulate `patents-public-data.patents.publications` table at *family_id* level"""
    query = f"""
     WITH pub AS (
    SELECT
      p.publication_number,
      publication_date,
      country_code,
      family_id,
      cpc.code as cpc_code,
      ipc.code as ipc_code,
      citation.publication_number as citation_publication_number
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST(cpc) AS cpc,
      UNNEST(ipc) AS ipc,
      UNNEST(citation) as citation
    )
    SELECT
      family_id,
      MIN(publication_date) AS publication_date,
      SPLIT(STRING_AGG(DISTINCT(p.publication_number))) AS publication_number,
      SPLIT(STRING_AGG(DISTINCT(country_code))) AS country_code,
      #SPLIT(STRING_AGG(DISTINCT(country))) AS country,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(pub.publication_number))) AS publication_number)] AS citation,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(cited_by.publication_number))) AS publication_number)] AS cited_by,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(cpc_code))) AS code)] AS cpc,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(ipc_code))) AS code)] AS ipc,
      ANY_VALUE(abstract) AS abstract
    FROM
      `patents-public-data.google_patents_research.publications` AS p,
       UNNEST(cited_by) as cited_by,
       pub
    WHERE
      pub.publication_number=p.publication_number
      AND p.abstract IS NOT NULL
      AND p.abstract !=""
    GROUP BY
      family_id
      """
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


@app.command()
def get_wgp_pubnum(
    flavor,
    table_wgp: str,
    table_tls211: str,
    destination_table: str,
    credentials: Path,
    verbose: bool = False,
):
    """
    Add a publication_number field in the geo table à la de Rassenfosse, Kozak and Seliger
    """
    # TODO Add family id at least. Ideally, deprecate in favor of WGP a la PatentCity
    query = f"""
    WITH app2pub AS (
        SELECT
          REPLACE(CONCAT(publn_auth, "-", publn_nr, "-", publn_kind), " ", "") AS publication_number,
          appln_id
        FROM
          {table_tls211}
        WHERE
          publn_nr IS NOT NULL
          AND publn_nr != ""
      )
        SELECT
          {flavor}.appln_id,
          app2pub.publication_number,
          patent_office,
          priority_date,
          city,
          lat,
          lng,
          name_0,
          name_1,
          name_2,
          name_3,
          name_4,
          name_5
        FROM
          {table_wgp} AS {flavor},
          app2pub
        WHERE
          app2pub.appln_id={flavor}.appln_id
    """
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


if __name__ == "__main__":
    app()

import typer
from pathlib import Path
from techlandscape.utils import get_bq_job_done
from techlandscape.decorators import monitor

app = typer.Typer()


@app.command()
@monitor
def get_publications_family(
    destination_table: str, credentials: Path, verbose: bool = False
):
    """Emulate `patents-public-data.patents.publications` table at the *family_id* level

    Arguments:
        destination_table: the BQ destination table (`project.dataset.table`)
        credentials: BQ credentials file path
        verbose: verbosity

    **Usage:**
        ```shell
        patentcity external get-publications-family <your-table> <your-credentials>
        ```

    !!! note
        It takes up to 2 minutes to complete

    """
    query = f"""
        WITH
          fam AS (
          SELECT
            DISTINCT family_id
          FROM
            `patents-public-data.patents.publications` ),
          crossover AS (
          SELECT
            publication_number,
            family_id
          FROM
            `patents-public-data.patents.publications` ),
          pub AS (
          SELECT
            family_id,
            MIN(publication_date) AS publication_date,
            SPLIT(STRING_AGG(DISTINCT(p.publication_number))) AS publication_number,
            SPLIT(STRING_AGG(DISTINCT(country_code))) AS country_code
          FROM
            `patents-public-data.patents.publications` AS p
          GROUP BY
            family_id ),
          tech_class AS (
          SELECT
            family_id,
            [STRUCT(SPLIT(STRING_AGG(DISTINCT(cpc.code))) AS code)] AS cpc,
            [STRUCT(SPLIT(STRING_AGG(DISTINCT(ipc.code))) AS code)] AS ipc
          FROM
            `patents-public-data.patents.publications` AS p,
            UNNEST(cpc) AS cpc,
            UNNEST(ipc) AS ipc
          GROUP BY
            family_id ),
          cit AS (
          SELECT
            family_id,
            [STRUCT(SPLIT(STRING_AGG(DISTINCT(citation.publication_number))) AS publication_number)] AS citation
          FROM
            `patents-public-data.patents.publications` AS p,
            UNNEST(citation) AS citation
          GROUP BY
            family_id ),
          gpr AS (
          SELECT
            family_id,
            [STRUCT(SPLIT(STRING_AGG(DISTINCT(cited_by.publication_number))) AS publication_number)] AS cited_by,
            ANY_VALUE(abstract) AS abstract
          FROM
            `patents-public-data.google_patents_research.publications` AS p,
            UNNEST(cited_by) AS cited_by
          LEFT JOIN
            crossover
          ON
            p.publication_number = crossover.publication_number
          GROUP BY
            family_id )
        SELECT
          fam.family_id,
          pub.* EXCEPT(family_id),
          tech_class.* EXCEPT(family_id),
          cit.* EXCEPT(family_id),
          gpr.* EXCEPT(family_id)
        FROM
          fam
        LEFT JOIN
          pub
        ON
          fam.family_id = pub.family_id
        LEFT JOIN
          tech_class
        ON
          fam.family_id = tech_class.family_id
        LEFT JOIN
          cit
        ON
          fam.family_id = cit.family_id
        LEFT JOIN
          gpr
        ON
          fam.family_id = gpr.family_id
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

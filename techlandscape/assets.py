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
        techlandscape assets get-publications-family <your-table> <your-credentials>
        ```

    !!! note
        It takes up to 2 minutes to complete

    !!! warning
        This table does not include ALL abstracts (due to ANY_VALUE considering NULL values as well).
        If you need to get all abstracts available, build abstract family table using `get_abstract_family`.

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
              p.family_id,
              [STRUCT(SPLIT(STRING_AGG(DISTINCT(crossover.family_id))) AS family_id)] AS citation
            FROM
              `patents-public-data.patents.publications` AS p,
              UNNEST(citation) AS citation
              LEFT JOIN 
                crossover 
                ON citation.publication_number = crossover.publication_number
            GROUP BY
              p.family_id),
            tmp_gpr AS (
              SELECT
                family_id,
                SPLIT(STRING_AGG(DISTINCT(cited_by.publication_number))) AS cited_by_publication_number,
                CONCAT(ANY_VALUE(title), "\\n", ANY_VALUE(abstract)) AS abstract
                #ANY_VALUE(abstract) AS abstract
              FROM
                `patents-public-data.google_patents_research.publications` AS p,
                UNNEST(cited_by) AS cited_by
              LEFT JOIN
                crossover
              ON
                p.publication_number = crossover.publication_number
              GROUP BY
                family_id),
              gpr AS (
              SELECT
                tmp_gpr.family_id,
                ANY_VALUE(abstract) AS abstract,
                [STRUCT(SPLIT(STRING_AGG(DISTINCT(crossover.family_id))) AS family_id)] AS cited_by
                #SPLIT(STRING_AGG(DISTINCT(cited_by_publication_number))) AS publication_number)] AS cited_by
            FROM
              tmp_gpr,
              UNNEST(cited_by_publication_number) AS cited_by_publication_number
            LEFT JOIN
              crossover
            ON
              cited_by_publication_number = crossover.publication_number
            GROUP BY
              tmp_gpr.family_id)
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
@monitor
def get_family_abstract(
    destination_table: str, credentials: Path, verbose: bool = False
):
    """
    Return a table at the family level. Specific attention is on making sure that alla families having at least one
    non NULL abstract have indeed a non NULL abstract. Here we do not draw any value from the all patents in the
    families, as is done, for the sake of simplicity, in `get_publications_family`. Instead, we first filter out patents
    with a NULL or empty abstract, and then only, we group by

    Arguments:
        destination_table: the BQ destination table (`project.dataset.table`)
        credentials: BQ credentials file path
        verbose: verbosity

    **Usage:**
        ```shell
        techlandscape assets get-family-abstract patentcity.techdiffusion.family_abstract credentials_bs.json
        ```
    """
    query = """
    WITH
      family_list AS(
      SELECT
        family_id,
        STRING_AGG(publication_number) AS publication_number
      FROM
        `patents-public-data.patents.publications`
      GROUP BY
        family_id ),
      tmp AS (
      SELECT
        publication_number,
        abstract
      FROM
        `patents-public-data.google_patents_research.publications` AS gpr
      WHERE
        abstract IS NOT NULL
        AND abstract !=""),
      family_with_abstract AS (
      SELECT
        p.family_id,
        ANY_VALUE(abstract) AS abstract
      FROM
        tmp
      LEFT JOIN
        `patents-public-data.patents.publications` AS p
      ON
        p.publication_number=tmp.publication_number
      GROUP BY
        p.family_id )
    SELECT
      fl.family_id,
      fl.publication_number,
      fwa.abstract,
      abstract IS NOT NULL
      AND abstract !="" AS has_abstract
    FROM
      family_with_abstract AS fwa
    RIGHT JOIN
      family_list AS fl
    ON
      fwa.family_id = fl.family_id"""
    get_bq_job_done(query, destination_table, credentials, verbose=verbose)


@app.command()
def _get_wgp_pubnum(
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

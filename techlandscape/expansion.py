from pathlib import Path
import typer
from techlandscape.utils import get_bq_job_done, get_project_id, get_uid

app = typer.Typer()


def get_seed_pc_freq(
    pc_flavor: str,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    key: str = "publication_number",
    **kwargs,
):
    """
    Return the frequency of patent classes in the seed

    Values:
        - pc_flavors: "cpc", "ipc"
        - key: "publication_number", "family_id"
    """
    assert pc_flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    project_id = get_project_id(key, credentials)

    query = f"""
    WITH tmp AS (
          SELECT
            tmp.{key},
            STRING_AGG(DISTINCT({pc_flavor}.code)) AS {pc_flavor},
            FROM
              `{project_id}.patents.publications` AS p,
              `{table_ref}` AS tmp,
              UNNEST({pc_flavor}) AS {pc_flavor}
            WHERE
              p.{key}=tmp.{key}
          GROUP BY
            {key}#, publication_year
            ),
            total AS (SELECT COUNT({key}) as count FROM
                `{table_ref}`
            )
    SELECT
          {pc_flavor},
          count(SPLIT({pc_flavor}, ",")) as n_{pc_flavor},
          total.count as n_patents,
          COUNT(SPLIT({pc_flavor}, ","))/total.count as freq
        FROM
          tmp,
          total
        GROUP BY
          {pc_flavor},
          total.count
        ORDER BY
          freq DESC 
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_universe_pc_freq(
    pc_flavor: str,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Return the frequency of patent classes in the universe of patents. Nb: we restrict to patents published between p25
    and p75 of the publication_year of patents in the seed.

    Values:
        - pc_flavors: "cpc", "ipc"
    """
    # TODO unrestricted time span option?
    # Support family_id level?
    assert pc_flavor in ["ipc", "cpc"]
    query = f"""
    WITH
      tmp AS (
      SELECT
        tmp.publication_number,
        ROUND(p.publication_date/10000, 0) AS publication_year
      FROM
        {table_ref} AS tmp,
        `patents-public-data.patents.publications` AS p
      WHERE
        p.publication_number=tmp.publication_number
        AND tmp.expansion_level="SEED"),
      stats AS (
      SELECT
        percentiles[OFFSET(25)] AS p25,
        percentiles[OFFSET(75)] AS p75
      FROM (
        SELECT
          APPROX_QUANTILES(publication_year, 100) AS percentiles
        FROM
          tmp)),
          total as (SELECT
          count(publication_number) as count
      FROM
      `patents-public-data.patents.publications` AS p,
      stats
    WHERE
      publication_date BETWEEN stats.p25*10000
      AND stats.p75*10000)
    SELECT
      {pc_flavor}.code AS {pc_flavor},
      COUNT({pc_flavor}.code) as n_cpc,
      total.count as n_patents,
      COUNT({pc_flavor}.code)/total.count as freq
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST({pc_flavor}) AS {pc_flavor},
      stats,
      total
    WHERE
      publication_date BETWEEN stats.p25*10000
      AND stats.p75*10000
    GROUP BY
      {pc_flavor},
      total.count
    ORDER BY
      freq DESC  
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_seed_pc_odds(
    pc_flavor: str,
    table_seed: str,
    table_universe: str,
    destination_table: str,
    credentials: Path,
    key="publication_number",
    **kwargs,
):
    """
    Return the odds of patent classes in seed compared to the universe of patents

    Values:
        - pc_flavors: "cpc", "ipc"
        - key: "publication_number", "family_id"
    """
    assert pc_flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    query = f"""
    WITH tmp AS (
        SELECT 
            seed.{pc_flavor} as {pc_flavor},
            seed.freq as seed_freq, 
            universe.freq as universe_freq
        FROM
            `{table_seed}` as seed,
            `{table_universe}` as universe
        WHERE seed.{pc_flavor} = universe.{pc_flavor} and seed.{pc_flavor} IS NOT NULL)
    SELECT
        {pc_flavor},
        seed_freq,
        universe_freq,
        seed_freq / universe_freq as odds
    FROM tmp
    ORDER BY 
        odds DESC 
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_pc_expansion(
    pc_flavor: str,
    n_pc: int,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    key="publication_number",
    **kwargs,
):
    """
    Expand the seed "along" the pc dimension for patent classes with large odds in

    Values:
        - pc_flavors: "cpc", "ipc"
        - key: "publication_number", "family_id"
    """
    assert pc_flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    project_id = get_project_id(key, credentials)

    query = f"""
    WITH
      important_pc AS (
      SELECT
        {pc_flavor}
      FROM
        `{table_ref}`
      ORDER BY
        odds DESC
      LIMIT
        {n_pc} ),
      patents AS (
      SELECT
        {key},
        {pc_flavor}.code AS {pc_flavor}
      FROM
        `{project_id}.patents.publications`,
        UNNEST({pc_flavor}) AS {pc_flavor} )
    SELECT
      patents.{key},
      "PC" as expansion_level
      # STRING_AGG(DISTINCT(patents.{pc_flavor})) AS {pc_flavor}  ## dbg
    FROM
      important_pc
    LEFT JOIN
      patents
    ON
      important_pc.{pc_flavor}=patents.{pc_flavor}
    GROUP BY
      {key}
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_citation_expansion(
    cit_flavor: str,
    expansion_level: str,
    table_ref: str,
    credentials: Path,
    key="publication_number",
    **kwargs,
):
    """
    Expand "along" the citation dimension, either backward or forward

    Values:
        - `cit_flavor`: "back", "for"
        - `key`:  "publication_number", "family_id"
    """
    assert expansion_level in ["L1", "L2"]
    assert cit_flavor in ["back", "for"]

    dataset = "patents" if cit_flavor == "back" else "google_patents_research"
    expansion_level_clause = (
        'AND expansion_level in ("SEED", "PC")'
        if expansion_level == "L1"
        else 'AND expansion_level LIKE "L1%"'
    )
    project_id_ = get_project_id(key, credentials)
    query_suffix = (
        "" if key == "publication_number" else f""", UNNEST({key}) as {key}"""
    )
    cit_var = "citation" if cit_flavor == "back" else "cited_by"

    query = f"""
    WITH expansion AS(
    SELECT
      cit.{key}
    FROM
      `{project_id_}.{dataset}.publications` AS p,
      {table_ref} AS tmp,
      UNNEST({cit_var}) AS cit
    WHERE
      p.{key}=tmp.{key}
      AND cit.{key} IS NOT NULL
      {expansion_level_clause}
    )
    SELECT 
      DISTINCT({key}),
      "{'-'.join([expansion_level, cit_flavor.upper()])}" AS expansion_level
    FROM
      expansion{query_suffix}      
    """
    destination_table = kwargs.get("destination_table", table_ref)
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_full_citation_expansion(
    expansion_level: str,
    table_ref: str,
    credentials: Path,
    key="publication_number",
    **kwargs,
):
    """Expand along the citation level, both backward and forward

    Values:
        - `expansion_level`: "L1", "L2"
        - `key`:  "publication_number", "family_id"
    """
    assert expansion_level in ["L1", "L2"]
    get_citation_expansion(
        "back", expansion_level, table_ref, credentials, key, **kwargs
    )
    get_citation_expansion(
        "for", expansion_level, table_ref, credentials, key, **kwargs
    )


@app.command()
def get_expansion(
    pc_flavor: str,
    table_ref: str,
    staging_dataset: str,
    credentials: Path,
    key: str = "publication_number",
    n_pc: int = 50,
    **kwargs,
):
    """
    Expand along PC and citations (L1 + L2)
    """
    name = table_ref.split(".")[-1]
    get_seed_pc_freq(
        pc_flavor,
        table_ref,
        f"{staging_dataset}.{name}_seed_{pc_flavor}_freq",
        credentials,
        key,
        **kwargs,
    )
    get_universe_pc_freq(
        pc_flavor,
        table_ref,
        f"{staging_dataset}.{name}_universe_{pc_flavor}_freq",
        credentials,
        **kwargs,
    )
    get_seed_pc_odds(
        pc_flavor,
        f"{staging_dataset}.{name}_seed_{pc_flavor}_freq",
        f"{staging_dataset}.{name}_universe_{pc_flavor}_freq",
        f"{staging_dataset}.{name}_seed_{pc_flavor}_odds",
        credentials,
        key,
        **kwargs,
    )
    get_pc_expansion(
        pc_flavor,
        n_pc,
        f"{staging_dataset}.{name}_seed_{pc_flavor}_odds",
        table_ref,
        credentials,
        key,
        write_disposition="WRITE_APPEND",
        **kwargs,
    )
    get_full_citation_expansion(
        "L1",
        table_ref,
        credentials,
        key,
        write_disposition="WRITE_APPEND",
        **kwargs,
    )
    get_full_citation_expansion(
        "L2",
        table_ref,
        credentials,
        key,
        write_disposition="WRITE_APPEND",
        **kwargs,
    )

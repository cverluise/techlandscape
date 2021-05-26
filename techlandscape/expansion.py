from pathlib import Path
import typer
from techlandscape.utils import get_bq_job_done, get_project_id
from techlandscape.enumerators import (
    TechClass,
    PrimaryKey,
    CitationKind,
    CitationExpansionLevel,
)

app = typer.Typer()


def get_seed_pc_freq(
    tech_class: TechClass,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    **kwargs,
):
    """
    Return the frequency of patent classes in the seed

    Arguments:
        tech_class:
        table_ref:
        destination_table:
        credentials:
        primary_key:
        **kwargs:

    **Usage:**

    """

    project_id = get_project_id(primary_key.value, credentials)

    query = f"""
    WITH tmp AS (
          SELECT
            tmp.{primary_key.value},
            STRING_AGG(DISTINCT({tech_class.value}.code)) AS {tech_class.value},
            FROM
              `{project_id}.patents.publications` AS p,
              `{table_ref}` AS tmp,
              UNNEST({tech_class.value}) AS {tech_class.value}
            WHERE
              p.{primary_key.value}=tmp.{primary_key.value}
          GROUP BY
            {primary_key.value}#, publication_year
            ),
            total AS (SELECT COUNT({primary_key.value}) as count FROM
                `{table_ref}`
            )
    SELECT
          {tech_class.value},
          count(SPLIT({tech_class.value}, ",")) as n_{tech_class.value},
          total.count as n_patents,
          COUNT(SPLIT({tech_class.value}, ","))/total.count as freq
        FROM
          tmp,
          total
        GROUP BY
          {tech_class.value},
          total.count
        ORDER BY
          freq DESC 
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_universe_pc_freq(
    tech_class: TechClass,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Return the frequency of patent classes in the universe of patents. Nb: we restrict to patents published between p25
    and p75 of the publication_year of patents in the seed.

    Arguments:
        tech_class:
        table_ref:
        destination_table:
        credentials:
        **kwargs:

    **Usage:**
    """
    # TODO unrestricted time span option?
    # Support family_id level?

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
      {tech_class.value}.code AS {tech_class.value},
      COUNT({tech_class.value}.code) as n_cpc,
      total.count as n_patents,
      COUNT({tech_class.value}.code)/total.count as freq
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST({tech_class.value}) AS {tech_class.value},
      stats,
      total
    WHERE
      publication_date BETWEEN stats.p25*10000
      AND stats.p75*10000
    GROUP BY
      {tech_class.value},
      total.count
    ORDER BY
      freq DESC  
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_seed_pc_odds(
    tech_class: TechClass,
    table_seed: str,
    table_universe: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Return the odds of patent classes in seed compared to the universe of patents

    Arguments:
        tech_class:
        table_seed:
        table_universe:
        destination_table:
        credentials:
        **kwargs:

    """

    query = f"""
    WITH tmp AS (
        SELECT 
            seed.{tech_class.value} as {tech_class.value},
            seed.freq as seed_freq, 
            universe.freq as universe_freq
        FROM
            `{table_seed}` as seed,
            `{table_universe}` as universe
        WHERE seed.{tech_class.value} = universe.{tech_class.value} and seed.{tech_class.value} IS NOT NULL)
    SELECT
        {tech_class.value},
        seed_freq,
        universe_freq,
        seed_freq / universe_freq as odds
    FROM tmp
    ORDER BY 
        odds DESC 
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_pc_expansion(
    tech_class: TechClass,
    n_pc: int,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    **kwargs,
):
    """
    Expand the seed "along" the pc dimension for patent classes with large odds in

    Arguments:
        tech_class:
        n_pc:
        table_ref:
        destination_table:
        credentials:
        primary_key:
        **kwargs:

    **Usage:**
    """

    project_id = get_project_id(primary_key, credentials)

    query = f"""
    WITH
      important_pc AS (
      SELECT
        {tech_class.value}
      FROM
        `{table_ref}`
      ORDER BY
        odds DESC
      LIMIT
        {n_pc} ),
      patents AS (
      SELECT
        {primary_key.value},
        {tech_class.value}.code AS {tech_class.value}
      FROM
        `{project_id}.patents.publications`,
        UNNEST({tech_class.value}) AS {tech_class.value} )
    SELECT
      patents.{primary_key.value},
      "PC" as expansion_level
      # STRING_AGG(DISTINCT(patents.{tech_class.value})) AS {tech_class.value}  ## dbg
    FROM
      important_pc
    LEFT JOIN
      patents
    ON
      important_pc.{tech_class.value}=patents.{tech_class.value}
    GROUP BY
      {primary_key.value}
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_full_pc_expansion(
    tech_class: TechClass,
    n_pc: int,
    table_ref: str,
    staging_dataset: str,
    credentials: Path,
    precomputed: bool = False,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    **kwargs,
):
    """Compute (or use precomputed) seed pc odds and expand along the pc dimension.

    Arguments:
        tech_class:
        n_pc:
        table_ref:
        staging_dataset:
        credentials:
        precomputed:
        primary_key:
        **kwargs:

    **Usage:**
    """
    name = table_ref.split(".")[-1]

    if not precomputed:
        get_seed_pc_freq(
            tech_class,
            table_ref,
            f"{staging_dataset}.{name}_seed_{tech_class.value}_freq",
            credentials,
            primary_key,
            verbose=False,
        )
        get_universe_pc_freq(
            tech_class,
            table_ref,
            f"{staging_dataset}.{name}_universe_{tech_class.value}_freq",
            credentials,
            verbose=False,
        )
        get_seed_pc_odds(
            tech_class,
            f"{staging_dataset}.{name}_seed_{tech_class.value}_freq",
            f"{staging_dataset}.{name}_universe_{tech_class.value}_freq",
            f"{staging_dataset}.{name}_seed_{tech_class.value}_odds",
            credentials,
            # primary_key,
            verbose=False,
        )
    get_pc_expansion(
        tech_class,
        n_pc,
        f"{staging_dataset}.{name}_seed_{tech_class.value}_odds",
        table_ref,
        credentials,
        primary_key,
        **kwargs,
    )


def get_citation_expansion(
    citation_kind: CitationKind,
    expansion_level: CitationExpansionLevel,
    table_ref: str,
    credentials: Path,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    **kwargs,
):
    """
    Expand "along" the citation dimension, either backward or forward

    Values:
        - `citation_kind`: "back", "for"
        - `primary_key`:  "publication_number", "family_id"
    """
    assert citation_kind in ["back", "for"]

    dataset = (
        "patents"
        if citation_kind.value == CitationKind.backward.value
        else "google_patents_research"
    )
    expansion_level_clause = (
        'AND expansion_level in ("SEED", "PC")'
        if expansion_level.value == CitationExpansionLevel.L1.value
        else 'AND expansion_level LIKE "L1%"'
    )
    project_id_ = get_project_id(primary_key, credentials)
    query_suffix = (
        ""
        if primary_key.value == PrimaryKey.publication_number.value
        else f""", UNNEST({primary_key.value}) as {primary_key.value}"""
    )
    cit_var = (
        "citation" if citation_kind.value == CitationKind.backward.value else "cited_by"
    )

    query = f"""
    WITH expansion AS(
    SELECT
      cit.{primary_key.value}
    FROM
      `{project_id_}.{dataset}.publications` AS p,
      {table_ref} AS tmp,
      UNNEST({cit_var}) AS cit
    WHERE
      p.{primary_key.value}=tmp.{primary_key.value}
      AND cit.{primary_key.value} IS NOT NULL
      {expansion_level_clause}
    )
    SELECT 
      DISTINCT({primary_key.value}),
      "{'-'.join([expansion_level.value, citation_kind.upper()])}" AS expansion_level
    FROM
      expansion{query_suffix}      
    """
    destination_table = kwargs.get("destination_table", table_ref)
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_full_citation_expansion(
    expansion_level: CitationExpansionLevel,
    table_ref: str,
    credentials: Path,
    primary_key: PrimaryKey = PrimaryKey.publication_number,
    **kwargs,
):
    """Expand along the citation level, both backward and forward

    Arguments:
        expansion_level:
        table_ref:
        credentials:
        primary_key:
        **kwargs:

    **Usage:**
    """
    get_citation_expansion(
        CitationKind.backward,
        expansion_level,
        table_ref,
        credentials,
        primary_key,
        **kwargs,
    )
    get_citation_expansion(
        CitationKind.forward,
        expansion_level,
        table_ref,
        credentials,
        primary_key,
        **kwargs,
    )


@app.command()
def get_expansion(
    table_ref: str,
    staging_dataset: str,
    credentials: Path,
    primary_key: PrimaryKey,
    tech_class: TechClass,
    precomputed: bool = False,
    n_pc: int = 50,
):
    """
    Expand along PC and citations (L1 + L2)

    Arguments:
        table_ref:
        staging_dataset:
        credentials:
        primary_key:
        tech_class:
        precomputed:
        n_pc:
    """

    get_full_pc_expansion(
        tech_class,
        n_pc,
        table_ref,
        staging_dataset,
        credentials,
        precomputed,
        primary_key,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )
    get_full_citation_expansion(
        CitationExpansionLevel.L1,
        table_ref,
        credentials,
        primary_key,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )
    get_full_citation_expansion(
        CitationExpansionLevel.L2,
        table_ref,
        credentials,
        primary_key,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )


if __name__ == "__main__":
    app()

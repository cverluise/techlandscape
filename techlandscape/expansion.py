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


def get_seed(
    primary_key: PrimaryKey,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    random_share: float = None,
    verbose: bool = False,
    **kwargs,
):
    """
    Return the starting block of the expansion. Random draw enabled (e.g. for robustness analysis)

    Arguments:
        primary_key: table primary key
        table_ref: manually annotated seed table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        random_share: size of the random draw (if not None)
        verbose: verbosity
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**

    """
    random_share_clause = f"""AND RAND()<{random_share}""" if random_share else ""

    query = f"""
    SELECT 
      {primary_key.value},
      expansion_level
    FROM 
      `{table_ref}`  # patentcity.techdiffusion.seed_additivemanufacturing
    WHERE 
      expansion_level LIKE "%SEED%"
      {random_share_clause}"""
    get_bq_job_done(query, destination_table, credentials, verbose=verbose, **kwargs)


def get_seed_pc_freq(
    primary_key: PrimaryKey,
    tech_class: TechClass,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Return the frequency of patent classes in the seed

    Arguments:
        primary_key: table primary key
        tech_class: technological class considered
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**

    """

    project_id = get_project_id(primary_key, credentials)

    query = f"""
    WITH tmp AS (
        SELECT
          tmp.{primary_key.value},
          {tech_class.value}.code AS {tech_class.value}
          FROM
            `{project_id}.patents.publications` AS p,
            `{table_ref}` AS tmp,
            UNNEST({tech_class.value}) AS {tech_class.value}
          WHERE
            p.{primary_key.value}=tmp.{primary_key.value}
            and tmp.expansion_level = "SEED"
        )
        ,
        total AS (
        SELECT 
          COUNT({primary_key.value}) as count
        FROM
          `{table_ref}`
        WHERE expansion_level = "SEED"
        )
        SELECT
          {tech_class.value},
          count({tech_class.value}) as n_{tech_class.value},
          total.count as n_patents,
          count({tech_class.value})/total.count as freq
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
    primary_key: PrimaryKey,
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
        primary_key: table primary key
        tech_class: technological class considered
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**
    """
    # TODO unrestricted time span option?
    # Support family_id level?
    project_id = get_project_id(primary_key, credentials)

    query = f"""
    WITH
      tmp AS (
      SELECT
        tmp.{primary_key.value},
        CAST(p.publication_date/10000 AS INT64) AS publication_year
      FROM
        {table_ref} AS tmp,
        `{project_id}.patents.publications` AS p
      WHERE
        p.{primary_key.value}=tmp.{primary_key.value}
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
          count({primary_key.value}) as count
      FROM
      `{project_id}.patents.publications` AS p,
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
      `{project_id}.patents.publications` AS p,
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
        tech_class: technological class considered
        table_seed: pc freq seed table (project.dataset.table)
        table_universe: pc freq universe table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

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
    primary_key: PrimaryKey,
    tech_class: TechClass,
    n_pc: int,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Expand the seed "along" the pc dimension for patent classes with large odds in

    Arguments:
        primary_key: table primary key
        tech_class: technological class considered
        n_pc: nb most important pc for pc expansion
        table_ref: pc odds table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

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
    primary_key: PrimaryKey,
    tech_class: TechClass,
    n_pc: int,
    table_ref: str,
    destination_table: str,
    staging_dataset: str,
    credentials: Path,
    precomputed: bool = False,
    **kwargs,
):
    """Compute (or use precomputed) seed pc odds and expand along the pc dimension.

    Arguments:
        primary_key: table primary key
        tech_class: technological class considered
        n_pc: nb most important pc for pc expansion
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        staging_dataset: intermediary table staging dataset (project.dataset)
        credentials: BQ credentials file path
        precomputed: True if pc odds pre-computed
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**
    """
    name = table_ref.split(".")[-1].replace("seed_", "")

    if not precomputed:
        get_seed_pc_freq(
            primary_key,
            tech_class,
            table_ref,
            f"{staging_dataset}.{name}_seed_{tech_class.value}_freq",
            credentials,
            verbose=False,
        )
        get_universe_pc_freq(
            primary_key,
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
            verbose=False,
        )
    get_pc_expansion(
        primary_key,
        tech_class,
        n_pc,
        f"{staging_dataset}.{name}_seed_{tech_class.value}_odds",
        destination_table,
        credentials,
        **kwargs,
    )


def get_citation_expansion(
    primary_key: PrimaryKey,
    citation_kind: CitationKind,
    expansion_level: CitationExpansionLevel,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Expand "along" the citation dimension, either backward or forward

    Arguments:
        primary_key: table primary key
        citation_kind: kind of citations
        expansion_level: expansion level
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**
    """

    dataset = (
        "google_patents_research"
        if citation_kind.value == CitationKind.forward.value
        and primary_key.value == PrimaryKey.publication_number.value
        else "patents"
    )
    expansion_level_clause = (
        'AND expansion_level in ("SEED", "PC")'
        if expansion_level.value == CitationExpansionLevel.L1.value
        else 'AND expansion_level LIKE "L1%"'
    )
    project_id_ = get_project_id(primary_key, credentials)
    # query_suffix = (
    #     ""
    #     if primary_key.value == PrimaryKey.publication_number.value
    #     else f""", UNNEST({primary_key.value}) as {primary_key.value}"""
    # )
    cit_var = (
        "citation" if citation_kind.value == CitationKind.backward.value else "cited_by"
    )

    query = f"""
    WITH expansion AS(
    SELECT
      DISTINCT cit.{primary_key.value}
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
      "{'-'.join([expansion_level.value, citation_kind.value.upper()])}" AS expansion_level
    FROM
      expansion      
    """
    # {query_suffix}
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_reduced_expansion(
    primary_key: PrimaryKey,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """
    Return the `table_ref` in a reduced format. Each `primary_key` appears only once

        primary_key: table primary key
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**
    """
    query = f"""
    SELECT 
    {primary_key.value},
    CASE
        WHEN STRING_AGG(expansion_level) LIKE "%ANTISEED%" THEN "ANTISEED" 
        WHEN STRING_AGG(expansion_level) LIKE "%SEED%" THEN "SEED"
        WHEN STRING_AGG(expansion_level) LIKE "%PC%" THEN "PC"
        WHEN STRING_AGG(expansion_level) LIKE "%L1-BACK%" THEN "L1-BACK"
        WHEN STRING_AGG(expansion_level) LIKE "%L1-FOR%" THEN "L1-FOR"
        WHEN STRING_AGG(expansion_level) LIKE "%L2-BACK%" THEN "L2-BACK"
        WHEN STRING_AGG(expansion_level) LIKE "%L2-FOR%" THEN "L2-FOR"
        ELSE NULL
    END AS expansion_level,
    STRING_AGG(expansion_level) as expansion_level_,
    COUNT(expansion_level) as nb_match,
    FROM `{table_ref}`
    GROUP BY {primary_key.value}
    """
    get_bq_job_done(query, destination_table, credentials, **kwargs)


def get_full_citation_expansion(
    primary_key: PrimaryKey,
    expansion_level: CitationExpansionLevel,
    table_ref: str,
    destination_table: str,
    credentials: Path,
    **kwargs,
):
    """Expand along the citation level, both backward and forward

    Arguments:
        primary_key: table primary key
        expansion_level: expansion level
        table_ref: expansion table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        credentials: BQ credentials file path
        **kwargs: key worded args passed to bigquery.QueryJobConfig

    **Usage:**
    """
    get_citation_expansion(
        primary_key,
        CitationKind.backward,
        expansion_level,
        table_ref,
        destination_table,
        credentials,
        **kwargs,
    )
    get_citation_expansion(
        primary_key,
        CitationKind.forward,
        expansion_level,
        table_ref,
        destination_table,
        credentials,
        **kwargs,
    )


@app.command()
def get_expansion(
    primary_key: PrimaryKey,
    tech_class: TechClass,
    table_ref: str,
    destination_table: str,
    staging_dataset: str,
    credentials: Path,
    random_share: float = None,
    precomputed: bool = False,
    n_pc: int = 50,
):
    """
    Expand along PC and citations (L1 + L2)

    Arguments:
        primary_key: table primary key
        tech_class: technological class considered
        table_ref: seed table (project.dataset.table)
        destination_table: query results destination table (project.dataset.table)
        staging_dataset: intermediary table staging dataset (project.dataset)
        credentials: BQ credentials file path
        random_share: share of the seed randomly drawn and used for the expansion
        precomputed: True if pc odds pre-computed
        n_pc: nb most important pc for pc expansion

    **Usage:**
        ```shell
        TECH="additivemanufacturing"
        techlandscape expansion get-expansion family_id cpc patentcity.techdiffusion.seed_${TECH} patentcity.techdiffusion.expansion_${TECH} patentcity.stage credentials_bq.json
        ```
    """

    get_seed(
        primary_key,
        table_ref,
        destination_table,
        credentials,
        random_share,
        write_disposition="WRITE_TRUNCATE",
        verbose=False,
    )

    get_full_pc_expansion(
        primary_key,
        tech_class,
        n_pc,
        destination_table,
        destination_table,
        staging_dataset,
        credentials,
        precomputed,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )
    get_full_citation_expansion(
        primary_key,
        CitationExpansionLevel.L1,
        destination_table,
        destination_table,
        credentials,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )
    get_full_citation_expansion(
        primary_key,
        CitationExpansionLevel.L2,
        destination_table,
        destination_table,
        credentials,
        write_disposition="WRITE_APPEND",
        verbose=False,
    )
    get_reduced_expansion(
        primary_key,
        destination_table,
        destination_table,
        credentials,
        write_disposition="WRITE_TRUNCATE",
        verbose=False,
    )


if __name__ == "__main__":
    app()

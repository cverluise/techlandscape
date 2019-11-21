from collections import Counter

import pandas as pd

from techlandscape.decorators import monitor, timer
from techlandscape.utils import format_table_ref_for_bq, flatten


# TODO control queries with publication_number=publication_number to check that inner join
#   rather than RIGHT/LEFT inner join is not detrimental


def _get_seed_pc_freq(flavor, client, table_ref):
    """
    Return a dataframe with the frequency of technological classes in the seed
    :param flavor: str, in ["ipc", "cpc"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pd.DataFrame
    """
    assert flavor in ["ipc", "cpc"]
    query = f"""
      SELECT
        tmp.publication_number,
        STRING_AGG({flavor}.code) AS {flavor}
        FROM
          `patents-public-data.patents.publications` AS p,
          {format_table_ref_for_bq(table_ref)} AS tmp,
          UNNEST({flavor}) AS {flavor}
        WHERE
          p.publication_number=tmp.publication_number
      GROUP BY
        publication_number
    """
    seed_pc = client.query(query=query).to_dataframe()[flavor].to_list()
    seed_nb_pat = len(seed_pc)
    # seed_pc is a list of comma separated strings. E.g. "A45D20/12,A45D20/122"
    seed_pc_list = flatten(list(map(lambda x: x.split(","), seed_pc)))
    # split strings into lists [['','',...],['','',...],..]
    # flatten the output ['','',...]
    seed_pc_count = Counter(seed_pc_list)
    # count the number of occurences of each code {'A':1, 'B':3, ...}
    # seed_pc_freq = seed_pc_count / seed_nb_pat
    return pd.DataFrame(
        index=seed_pc_count.keys(),
        data=list(map(lambda x: x / seed_nb_pat, seed_pc_count.values())),
        columns=["freq"],
    )


def _get_universe_pc_freq(flavor, client, table_ref):
    """
    Return the frequency of patent classes for patents with publication_year between p25 and p75
    of the publication_year of patents in the seed.
    :param flavor: str
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return:
    """
    assert flavor in ["ipc", "cpc"]
    query = f"""
    WITH
      tmp AS (
      SELECT
        tmp.publication_number,
        ROUND(p.publication_date/10000, 0) AS publication_year
      FROM
        {format_table_ref_for_bq(table_ref)} AS tmp,
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
      {flavor}.code AS {flavor},
      COUNT({flavor}.code) as count,
      total.count AS total
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST({flavor}) AS {flavor},
      stats,
      total
    WHERE
      publication_date BETWEEN stats.p25*10000
      AND stats.p75*10000
    GROUP BY
      {flavor}, total
    """

    df = client.query(query).to_dataframe()
    df = df.set_index(flavor)
    return (df["count"] / df["total"]).rename("freq").to_frame()


@timer
@monitor
def get_important_pc(flavor, threshold, client, table_ref):
    """
    :param flavor: str
    :param threshold: float, threshold above which a pc is considered over-represented in the
    seed wrt the universe
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return:
    """
    assert flavor in ["ipc", "cpc"]
    seed_pc_freq = _get_seed_pc_freq(flavor, client, table_ref)
    universe_pc_freq = _get_universe_pc_freq(flavor, client, table_ref)
    pc_odds = seed_pc_freq.merge(
        universe_pc_freq,
        right_index=True,
        left_index=True,
        suffixes=["_seed", "_universe"],
    )
    pc_odds["odds"] = pc_odds["freq_seed"] / pc_odds["freq_universe"]
    return pc_odds.query(
        "odds>@threshold"
    )  # list(cpc_odds.query("odds>@threshold").index)


@timer
@monitor
def pc_expansion(flavor, pc_list, client, job_config):
    """
    Expands the seed "along" the pc dimension for patent classes in <pc_list>
    :param flavor: str, in ["ipc", "cpc"]
    :param pc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    assert isinstance(pc_list, list)
    assert flavor in ["ipc", "cpc"]
    pc_string = ",".join(list(map(lambda x: '"' + x + '"', pc_list)))
    query = f"""
    SELECT
      DISTINCT(publication_number),
      "PC" as expansion_level
    FROM
      `patents-public-data.patents.publications`,
      UNNEST({flavor}) AS {flavor}
    WHERE
      {flavor}.code IN ( {pc_string} )
    """
    client.query(query, job_config=job_config).result()


@monitor
def citation_expansion(flavor, expansion_level, client, table_ref, job_config):
    """
    Expands "along" the citation dimension, either backward (flavor=="citation") or forward (
    citation=="cited_by")
    :param flavor: str, ["citation", "cited_by"]
    :param expansion_level: str, in ["citation", "cited_by"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    assert expansion_level in ["L1", "L2"]
    assert flavor in ["citation", "cited_by"]
    if flavor == "citation":
        dataset = "patents"
        suffix = "-BACK"
    else:
        dataset = "google_patents_research"
        suffix = "-FOR"
    if expansion_level == "L1":
        expansion_level_clause = 'AND expansion_level in ("SEED", "PC")'
    else:
        expansion_level_clause = 'AND expansion_level LIKE "L1%"'
    query = f"""
    SELECT
      DISTINCT(SPLIT({flavor}.publication_number)[OFFSET(0)]) AS publication_number,
      "{expansion_level + suffix}" AS expansion_level
    FROM
      `patents-public-data.{dataset}.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS tmp,
      UNNEST({flavor}) AS {flavor}
    WHERE
      p.publication_number=tmp.publication_number
      AND {flavor}.publication_number IS NOT NULL
      AND {flavor}.publication_number!=""
      {expansion_level_clause}
    """
    client.query(query, job_config=job_config).result()

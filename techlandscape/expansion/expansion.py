from collections import Counter
from typing import List
from pathlib import Path

import pandas as pd
import os
from wasabi import Printer

from techlandscape.decorators import monitor
from techlandscape.utils import (
    format_table_ref_for_bq,
    flatten,
    get_bq_job_done,
)
from techlandscape.query import get_pc_like_clause, get_project_id

# TODO control queries with publication_number=publication_number to check that inner join
#   rather than RIGHT/LEFT inner join is not detrimental
msg = Printer()


def _get_seed_pc_freq(
    flavor: str, client, table_ref: str, key: str = "publication_number"
):
    """
    Return a dataframe with the frequency of technological classes in the seed and their time range
    :param client: google.cloud.bigquery.client.Client
    :return: (pd.DataFrame, list), freq df, [yr_lo, yr_up]
    """
    assert flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    project_id_ = get_project_id(key, client)

    query = f"""
      SELECT
        tmp.{key},
        STRING_AGG({flavor}.code) AS {flavor},
        CAST(ROUND(p.publication_date/10000, 0) AS INT64) AS publication_year
        FROM
          `{project_id_}.patents.publications` AS p,
          {format_table_ref_for_bq(table_ref)} AS tmp,
          UNNEST({flavor}) AS {flavor}
        WHERE
          p.{key}=tmp.{key}
      GROUP BY
        {key}, publication_year
    """
    tmp = client.query(query=query).to_dataframe()
    time_range = list(
        map(
            lambda x: int(x),
            tmp["publication_year"].quantile([0.1, 0.9]).values,
        )
    )
    seed_pc = tmp[flavor].to_list()

    seed_nb_pat = len(seed_pc)
    # seed_pc is a list of comma separated strings. E.g. "A45D20/12,A45D20/122"
    seed_pc_list = flatten(list(map(lambda x: x.split(","), seed_pc)))
    # split strings into lists [['','',...],['','',...],..]
    # flatten the output ['','',...]
    seed_pc_count = Counter(seed_pc_list)
    # count the number of occurences of each code {'A':1, 'B':3, ...}
    # seed_pc_freq = seed_pc_count / seed_nb_pat
    return (
        pd.DataFrame(
            index=seed_pc_count.keys(),
            data=list(map(lambda x: x / seed_nb_pat, seed_pc_count.values())),
            columns=["freq"],
        ),
        time_range,
    )


def get_universe_pc_freq(
    flavor: str, table_ref: str, destination_table: str, credentials: Path
):
    """
    Return the frequency of patent classes of the universe of patents published between p25 and p75 of the
    publication_year of patents in the seed
    """
    # TODO unrestricted time span option?
    # TODO restricted countries option?
    assert flavor in ["ipc", "cpc"]
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
      {flavor}.code AS {flavor},
      #COUNT({flavor}.code) as count,
      #total.count AS total,
      COUNT({flavor}.code)/total.count as freq
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST({flavor}) AS {flavor},
      stats,
      total
    WHERE
      publication_date BETWEEN stats.p25*10000
      AND stats.p75*10000
    GROUP BY
      {flavor},
      total.count
    ORDER BY
      freq DESC  
    """
    get_bq_job_done(query, destination_table, credentials)


def _get_universe_pc_freq_from_file(f, flavor, time_range, countries=None):
    """
    Return the frequency of patent classes for patents with publication_year in <time_range>
    and country_code in <countries> if not None (no restriction if None)
    :param f: str, csv.gz file with universe pc freq (country_code|year|pc|freq)
    :param flavor: str, in ["cpc", "ipc"]
    :param time_range: List[int], [yr_lo, yr_up]
    :param countries: List[str], ISO2 countries we are interested in
    :return: pd.DataFrame
    """
    # TODO check that countries are ISO2, should be done earlier actually
    assert os.path.isfile(f)
    assert flavor in ["cpc", "ipc"]
    universe_pc_freq = pd.read_csv(f, index_col=0, compression="gzip")
    query = "@time_range[0]<=publication_year<=@time_range[1]"
    query = query + "  and country_code in @countries" if countries else query
    universe_pc_freq = universe_pc_freq.query(query)
    return universe_pc_freq.groupby(flavor).mean()["freq"]


@monitor
def get_important_pc(
    flavor,
    threshold,
    client,
    table_ref,
    key="publication_number",
    counterfactual_f=None,
    countries=None,
):
    """
    Return a dataframe with pc which are <threshold>-folds more represented in the seed than in the "universe"
    of patents (restr to <time_range> and <countries> if not None
    :param flavor: str
    :param threshold: float, threshold above which a pc is considered over-represented in the
    seed wrt the universe
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param key: str, in ["publication_number", "family_id"]
    :param counterfactual_f: str, csv.gz file with universe pc freq (country_code|year|pc|freq)
    :param countries: list, iso2 country list
    :return: pd.DataFrame
    """
    assert flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]
    seed_pc_freq, time_range = _get_seed_pc_freq(
        flavor, client, table_ref, key=key
    )
    if counterfactual_f:
        universe_pc_freq = _get_universe_pc_freq_from_file(
            counterfactual_f, flavor, time_range, countries
        )
        # Nb: all countries and years are equally weighted. Could be more sophisticated
    else:
        if key == "family_id":
            raise AttributeError(
                f"{key} mode requires a local counterfactual file. See bin/ExtractCntYrPC.py"
            )
        else:
            msg.info(
                "Local counterfactual file might considerably increase efficiency. "
                "Have a look at bin/ExtractCntYrPC.py"
            )
            universe_pc_freq = _get_universe_pc_freq(flavor, client, table_ref)
    pc_odds = seed_pc_freq.merge(
        universe_pc_freq,
        how="left",
        right_index=True,
        left_index=True,
        suffixes=["_seed", "_universe"],
    )
    pc_odds["odds"] = pc_odds["freq_seed"] / pc_odds["freq_universe"]
    pc_odds.loc[pc_odds["freq_universe"].isna(), "odds"] = threshold + 1
    # We keep cpcs with too few occurences to be kept in the local counterfactual file (below lower_bound)
    return pc_odds.query("odds>@threshold")
    # list(cpc_odds.query("odds>@threshold").index)


@monitor
def pc_expansion(
    flavor, pc_list, client, job_config, key="publication_number"
):
    """
    Expands the seed "along" the pc dimension for patent classes in <pc_list>
    :param flavor: str, in ["ipc", "cpc"]
    :param pc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :param key: str, in ["publication_number", "family_id"]
    :return:
    """
    assert isinstance(pc_list, list)
    assert flavor in ["ipc", "cpc"]
    assert key in ["publication_number", "family_id"]

    project_id_ = get_project_id(key, client)
    pc_like_clause_ = get_pc_like_clause(flavor, pc_list, sub_group=False)

    query = f"""
    SELECT
      DISTINCT({key}),
      "PC" as expansion_level
    FROM
      `{project_id_}.patents.publications`,
      UNNEST({flavor}) AS {flavor}
    WHERE
      {pc_like_clause_}
    """
    client.query(query, job_config=job_config).result()


def _citation_expansion(
    flavor,
    expansion_level,
    client,
    table_ref,
    job_config,
    key="publication_number",
):
    """
    Expands "along" the citation dimension, either backward (flavor=="citation") or forward (
    citation=="cited_by")
    :param flavor: str, ["citation", "cited_by"]
    :param expansion_level: str, in ["citation", "cited_by"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :param key: str, in ["publication_number", "family_id"]
    :return: bq.Job
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
    project_id_ = get_project_id(key, client)
    query_suffix = (
        "" if key == "publication_number" else f""", UNNEST({key}) as {key}"""
    )
    query = f"""
    WITH expansion AS(
    SELECT
      cit.{key}
    FROM
      `{project_id_}.{dataset}.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS tmp,
      UNNEST({flavor}) AS cit
    WHERE
      p.{key}=tmp.{key}
      AND cit.{key} IS NOT NULL
      {expansion_level_clause}
    )
    SELECT 
      DISTINCT({key}),
      "{expansion_level + suffix}" AS expansion_level
    FROM
      expansion{query_suffix}      
    """
    return client.query(query, job_config=job_config)


@monitor
def citation_expansion(
    expansion_level, client, table_ref, job_config, key="publication_number"
):
    back_job = _citation_expansion(
        "citation", expansion_level, client, table_ref, job_config, key
    )
    for_job = _citation_expansion(
        "cited_by", expansion_level, client, table_ref, job_config, key
    )

    for_job.result()
    back_job.result()

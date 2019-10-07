import os
from collections import Counter

import pandas as pd

from techlandscape.decorators import monitor
from techlandscape.utils import (
    format_table_ref_for_bq,
    country_clause_for_bq,
    flatten,
)


# TODO control queries with publication_number=publication_number to check that inner join
#   rather than RIGHT/LEFT inner join is not detrimental
# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning


@monitor
def load_seed_to_bq(f, client, table_ref, job_config):
    """

    :param f: str or pandas.core.frame.DataFrame
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.LoadJobConfig
    :return:
    """
    assert isinstance(f, (str, pd.DataFrame))
    if isinstance(f, str):
        assert os.path.exists(f)
        seed_df = pd.read_csv(f)
    else:
        assert "publication_number" in f.columns
        seed_df = f

    seed_df["expansion_level"] = "SEED"
    client.load_table_from_dataframe(
        seed_df, table_ref, job_config=job_config
    ).result()


def _get_seed_pc_freq(flavor, client, table_ref):
    """
    :param flavor:
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return:
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


@monitor
def get_important_cpc(flavor, threshold, client, table_ref):
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
    return pc_odds["odds"]  # list(cpc_odds.query("odds>@threshold").index)


@monitor
def pc_expansion(flavor, pc_list, client, job_config):
    """
    :param flavor: str
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
    client.query(query, job_config=job_config)


@monitor
def citation_expansion(expansion_level, client, table_ref, job_config):
    """

    :param expansion_level: str
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    assert expansion_level in ["L1", "L2"]
    if expansion_level == "L1":
        l1_clause = ""
    else:
        l1_clause = 'AND expansion_level="L1"'
    query = """
    SELECT
      DISTINCT(SPLIT(citation.publication_number)[OFFSET(0)]) AS publication_number,
      {} AS expansion_level
    FROM
      `patents-public-data.patents.publications` AS p,
      {} AS tmp,
      UNNEST(citation) AS citation
    WHERE
      p.publication_number=tmp.publication_number
      AND citation.publication_number IS NOT NULL
      AND citation.publication_number!=""
      {}
    """.format(
        '"' + expansion_level + '"',
        format_table_ref_for_bq(table_ref),
        l1_clause,
    )
    client.query(query, job_config=job_config)


@monitor
def draw_af_antiseed(size, client, table_ref, job_config):
    """

    :param size: int
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    query = """
SELECT
      DISTINCT(p.publication_number) AS publication_number,
      "ANTISEED-AF" AS expansion_level
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST(abstract_localized) as abstract
    LEFT OUTER JOIN
      {} AS tmp
    ON
      p.publication_number = tmp.publication_number
    WHERE
      p.country_code in ({})
      AND abstract.text is not NULL
      AND abstract.text!=''
      AND abstract.language="en"
    ORDER BY
      RAND()
    LIMIT
      1000
    """.format(
        format_table_ref_for_bq(table_ref), country_clause_for_bq(), size
    )
    client.query(query, job_config=job_config)


@monitor
def draw_aug_antiseed(size, flavor, pc_list, client, table_ref, job_config):
    """

    :param size: int
    :param flavor: str
    :param pc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    assert flavor in ["ipc", "cpc"]
    pc_like_clause = " OR ".join(
        set(
            list(
                map(
                    lambda x: f'{flavor}.code LIKE "' + x.split("/")[0] + '%"',
                    pc_list,
                )
            )
        )
    )
    query = f"""
    SELECT
      DISTINCT(p.publication_number) AS publication_number,
      "ANTISEED-AUG" AS expansion_level
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST({flavor}) AS {flavor},
      UNNEST(abstract_localized) AS abstract
    LEFT OUTER JOIN
      {format_table_ref_for_bq(table_ref)} AS tmp
    ON
      p.publication_number = tmp.publication_number
    WHERE
      {pc_like_clause}
      AND p.country_code in ({country_clause_for_bq()})
      AND abstract.text is not NULL
      AND abstract.text!=''
      AND abstract.language="en"
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    client.query(query, job_config=job_config)


@monitor
def get_expansion_result(client, table_ref):
    """
    Return the result of the expansion.
    E.g: publication_number | expansion_level | abstract
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pandas.core.frame.DataFrame
    """
    query = f"""
    SELECT
      tmp.publication_number,
      tmp.expansion_level,
      abstract.text as abstract
    FROM
      `patents-public-data.patents.publications` as p,
      {format_table_ref_for_bq(table_ref)} as tmp,
      UNNEST(abstract_localized) as abstract
    WHERE
      p.publication_number=tmp.publication_number
      AND p.country_code in ({country_clause_for_bq()})
      AND abstract.text is not NULL
      AND abstract.text!=''
      AND abstract.language="en"
    GROUP BY
      publication_number, expansion_level, abstract  
    """
    # nb: we could start by only loading the seed and anti seed to start the classification exercise
    # asap and load the rest of the dataset in the meantime
    # TODO add attributes country_code, assignee.name, assignee.country_code, inventor.name,
    #   inventor.country_code, publication_date, cpc, etc. Later location (from Gaëtan)
    #   -> for later, only on the class 0. NB:
    return client.query(query).to_dataframe()

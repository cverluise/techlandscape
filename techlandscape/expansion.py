import os
from collections import Counter

import pandas as pd

from techlandscape.decorators import monitor
from techlandscape.utils import (
    format_table_ref_for_bq,
    country_clause_for_bq,
    flatten,
)


# TODO: control queries with publication_number=publication_number to check that inner join
#   rather than RIGHT/LEFT inner join is not detrimental
# TODO: work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine-learning


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


def _get_seed_cpc_freq(client, table_ref):
    """

    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return:
    """
    query = """
    SELECT
      tmp.publication_number,
      STRING_AGG(cpc.code) AS cpc
      FROM
        `patents-public-data.patents.publications` AS p,
        {} AS tmp,
        UNNEST(cpc) AS cpc
      WHERE
        p.publication_number=tmp.publication_number
    GROUP BY
      publication_number
    """.format(
        format_table_ref_for_bq(table_ref)
    )
    seed_cpc = client.query(query=query).to_dataframe()["cpc"].to_list()
    seed_nb_pat = len(seed_cpc)
    # seed_cpc is a list of comma separated strings. E.g. "A45D20/12,A45D20/122"
    seed_cpc_list = flatten(list(map(lambda x: x.split(","), seed_cpc)))
    # split strings into lists [['','',...],['','',...],..]
    # flatten the output ['','',...]
    seed_cpc_count = Counter(seed_cpc_list)
    # count the number of occurences of each code {'A':1, 'B':3, ...}
    # seed_cpc_freq = seed_cpc_count / seed_nb_pat
    return pd.DataFrame(
        index=seed_cpc_count.keys(),
        data=list(map(lambda x: x / seed_nb_pat, seed_cpc_count.values())),
        columns=["freq"],
    )


def _get_universe_cpc_freq():
    assert os.path.exists("data/persist/cpc_counts.csv.gz")
    universe_cpc = pd.read_csv(
        "data/persist/cpc_counts.csv.gz", index_col=0, compression="gzip"
    )
    universe_cpc["freq"] = universe_cpc / universe_cpc.loc["nb_patents"].values
    return universe_cpc["freq"].to_frame()


@monitor
def get_important_cpc(threshold, client, table_ref):
    """

    :param threshold: float, threshold above which a cpc is considered over-represented in the
    seed wrt the universe
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return:
    """
    seed_cpc_freq = _get_seed_cpc_freq(client, table_ref)
    universe_cpc_freq = _get_universe_cpc_freq()
    cpc_odds = seed_cpc_freq.merge(
        universe_cpc_freq,
        right_index=True,
        left_index=True,
        suffixes=["_seed", "_universe"],
    )
    cpc_odds["odds"] = cpc_odds["freq_seed"] / cpc_odds["freq_universe"]
    return list(cpc_odds.query("odds>@threshold").index)


@monitor
def cpc_expansion(cpc_list, client, job_config):
    """

    :param cpc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    assert isinstance(cpc_list, list)
    cpc_string = ",".join(list(map(lambda x: '"' + x + '"', cpc_list)))
    query = """
    SELECT
      DISTINCT(publication_number),
      "CPC" as expansion_level
    FROM
      `patents-public-data.patents.publications`,
      UNNEST(cpc) AS cpc
    WHERE
      cpc.code IN ( {} )
    """.format(
        cpc_string
    )
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
      unnest(abstract_localized) as abstract
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
def draw_aug_antiseed(size, cpc_list, client, table_ref, job_config):
    """

    :param size: int
    :param cpc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    cpc_like_clause = " OR ".join(
        set(
            list(
                map(
                    lambda x: 'cpc.code LIKE "' + x.split("/")[0] + '%"',
                    cpc_list,
                )
            )
        )
    )
    query = """
    SELECT
      DISTINCT(p.publication_number) AS publication_number,
      "ANTISEED-AUG" AS expansion_level
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST(cpc) AS cpc
    LEFT OUTER JOIN
      {} AS tmp
    ON
      p.publication_number = tmp.publication_number
    WHERE
      {}
      AND p.country_code in ({})
      AND abstract.text is not NULL
      AND abstract.text!=''
      AND abstract.language="en"
    ORDER BY
      RAND()
    LIMIT
      {}
    """.format(
        format_table_ref_for_bq(table_ref),
        cpc_like_clause,
        country_clause_for_bq(),
        size,
    )
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
    query = """
    SELECT
      tmp.publication_number,
      tmp.expansion_level,
      abstract.text as abstract
    FROM
      `patents-public-data.patents.publications` as p,
      {} as tmp,
      UNNEST(abstract_localized) as abstract
    WHERE
      p.publication_number=tmp.publication_number
      AND p.country_code in ({})
      AND abstract.text is not NULL
      AND abstract.text!=''
      AND abstract.language="en"
    GROUP BY
      publication_number, expansion_level, abstract  
    """.format(
        format_table_ref_for_bq(table_ref), country_clause_for_bq()
    )
    # nb: we could start by only loading the seed and anti seed to start the classification exercise
    # asap and load the rest of the dataset in the meantime
    # TODO add attributes country_code, assignee.name, assignee.country_code, inventor.name,
    #   inventor.country_code, publication_date, cpc, etc. Later location (from Gaëtan)
    #   -> for later, only on the class 0. NB:
    return client.query(query).to_dataframe()

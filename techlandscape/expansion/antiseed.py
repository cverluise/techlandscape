from techlandscape.decorators import monitor, timer
from techlandscape.utils import (
    format_table_ref_for_bq,
    country_clause_for_bq,
    english_speaking_offices,
)

# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning


@timer
@monitor
def draw_af_antiseed(size, client, table_ref, job_config):
    """

    :param size: int
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
    """
    query = f"""
    SELECT
      DISTINCT(r.publication_number) AS publication_number,
      "ANTISEED-AF" AS expansion_level
    FROM
      `patents-public-data.google_patents_research.publications` AS r
    LEFT OUTER JOIN
      {format_table_ref_for_bq(table_ref)} AS tmp
    ON
      r.publication_number = tmp.publication_number
    WHERE
      r.country in ({country_clause_for_bq(english_speaking_offices)})
      AND r.abstract is not NULL
      AND r.abstract!=''
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    client.query(query, job_config=job_config)


@timer
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
    pc_like_clause = (
        "("
        + " OR ".join(
            set(
                list(
                    map(
                        lambda x: f'{flavor}.code LIKE "'
                        + x.split("/")[0]
                        + '%"',
                        pc_list,
                    )
                )
            )
        )
        + ")"
    )
    query = f"""
    SELECT
      DISTINCT(r.publication_number) AS publication_number,
      "ANTISEED-AUG" AS expansion_level
    FROM
      `patents-public-data.google_patents_research.publications` AS r,
      UNNEST({flavor}) AS {flavor}
    LEFT OUTER JOIN
      {format_table_ref_for_bq(table_ref)} AS tmp
    ON
      r.publication_number = tmp.publication_number
    WHERE
      {pc_like_clause}
      AND r.country in ({country_clause_for_bq(english_speaking_offices)})
      AND r.abstract is not NULL
      AND r.abstract!=''
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    client.query(query, job_config=job_config)

from techlandscape.decorators import monitor
from techlandscape.utils import format_table_ref_for_bq, country_clause_for_bq

# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning


def _draw_af_antiseed(size, client, table_ref, job_config, countries=None):
    """

    :param size: int
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return: bq.Job
    """
    country_clause = (
        f"AND r.country in ({country_clause_for_bq(countries)})"
        if countries
        else ""
    )
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
      r.abstract is not NULL
      AND r.abstract!=''
      {country_clause}
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    return client.query(query, job_config=job_config)


def _draw_aug_antiseed(
    size, flavor, pc_list, client, table_ref, job_config, countries=None
):
    """

    :param size: int
    :param flavor: str
    :param pc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return: bq.Job
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
    country_clause = (
        f"AND r.country in ({country_clause_for_bq(countries)})"
        if countries
        else ""
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
      {country_clause}
      AND r.abstract is not NULL
      AND r.abstract!=''
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    return client.query(query, job_config=job_config)


@monitor
def draw_antiseed(
    size, flavor, pc_list, client, table_ref, job_config, countries=None
):
    af_antiseed_job = _draw_af_antiseed(
        size, client, table_ref, job_config, countries
    )
    aug_antiseed_job = _draw_aug_antiseed(
        size, flavor, pc_list, client, table_ref, job_config, countries
    )
    af_antiseed_job.result()
    aug_antiseed_job.result()

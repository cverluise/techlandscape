from techlandscape.decorators import monitor
from techlandscape.utils import format_table_ref_for_bq
from techlandscape.expansion.utils import (
    country_clause,
    pc_like_clause,
    country_prefix,
    project_id,
)


# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning


def _draw_af_antiseed(
    size,
    client,
    table_ref,
    job_config,
    key="publication_number",
    countries=None,
):
    """
    Return the anti-seed a la AF
    :param size: int
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :param key: str, in ["publication_number", "family_id"]
    :param countries: List[str], ISO2 countries we are interested in
    :return: bq.Job
    """
    project_id_ = project_id(key, client)
    country_prefix_ = country_prefix(key)
    country_clause_ = country_clause(countries)
    query = f"""SELECT
      DISTINCT(r.{key}) AS {key},
      "ANTISEED-AF" AS expansion_level
    FROM
      `{project_id_}.google_patents_research.publications` AS r {country_prefix_}
    LEFT OUTER JOIN
      {format_table_ref_for_bq(table_ref)} AS tmp
    ON
      r.{key} = tmp.{key}
    WHERE
      r.abstract is not NULL
      AND r.abstract!=''
      {country_clause_}
    ORDER BY
      RAND()
    LIMIT
      {size}
    """
    return client.query(query, job_config=job_config)


def _draw_aug_antiseed(
    size,
    flavor,
    pc_list,
    client,
    table_ref,
    job_config,
    key="publication_number",
    countries=None,
):
    """
    Return the augmented anti-seed
    :param size: int
    :param flavor: str
    :param pc_list: list
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return: bq.Job
    """
    assert flavor in ["ipc", "cpc"]
    project_id_ = project_id(key, client)
    pc_like_clause_ = pc_like_clause(flavor, pc_list, sub_group=True)
    country_prefix_ = country_prefix(key)
    country_clause_ = country_clause(countries)
    query = f"""
    SELECT
      DISTINCT(r.{key}) AS {key},
      "ANTISEED-AUG" AS expansion_level
    FROM
      `{project_id_}.google_patents_research.publications` AS r,
      UNNEST({flavor}) AS {flavor} {country_prefix_}
    LEFT OUTER JOIN
      {format_table_ref_for_bq(table_ref)} AS tmp
    ON
      r.{key} = tmp.{key}
    WHERE
      {pc_like_clause_}
      {country_clause_}
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
    size,
    flavor,
    pc_list,
    client,
    table_ref,
    job_config,
    key="publication_number",
    countries=None,
):
    af_antiseed_job = _draw_af_antiseed(
        size, client, table_ref, job_config, key=key, countries=countries
    )
    aug_antiseed_job = _draw_aug_antiseed(
        size,
        flavor,
        pc_list,
        client,
        table_ref,
        job_config,
        key=key,
        countries=countries,
    )
    af_antiseed_job.result()
    aug_antiseed_job.result()

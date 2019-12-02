from techlandscape.decorators import monitor, timer
from techlandscape.utils import format_table_ref_for_bq, country_clause_for_bq
import asyncio


# TODO work on reproducibility when calling random draw
#   Could find some inspiration here
#   https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine
#   -learning


@timer
# @monitor
async def draw_af_antiseed(
    size, client, table_ref, job_config, countries=None
):
    """

    :param size: int
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param job_config: google.cloud.bigquery.job.QueryJobConfig
    :return:
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
    client.query(query, job_config=job_config).result()


@timer
# @monitor
async def draw_aug_antiseed(
    size, flavor, pc_list, client, table_ref, job_config, countries=None
):
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
    client.query(query, job_config=job_config).result()


async def draw_antiseed(
    size, flavor, pc_list, client, table_ref, job_config, countries=None
):
    af_antiseed_task = asyncio.create_task(
        draw_af_antiseed(size, client, table_ref, job_config, countries)
    )
    aug_antiseed_task = asyncio.create_task(
        draw_aug_antiseed(
            size,
            flavor,
            pc_list,  # we exclude _all_ important pcs
            client,
            table_ref,
            job_config,
            countries,
        )
    )
    await af_antiseed_task
    await aug_antiseed_task

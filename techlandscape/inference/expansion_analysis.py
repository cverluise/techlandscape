from techlandscape.utils import format_table_ref_for_bq
from techlandscape.decorators import monitor


@monitor
def count_expansion_level(client, table_ref):
    """
    Return the number of patents in each expansion_level (e.g: SEED, ANTISEED-XX, L1-XX, etc)
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pd.DataFrame
    """

    query = f"""
    SELECT
      expansion_level,
      COUNT(expansion_level) as nb_patent
    FROM
      {format_table_ref_for_bq(table_ref)}
    GROUP BY
      expansion_level
    """
    return client.query(query).to_dataframe()

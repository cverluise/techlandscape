from techlandscape.utils import format_table_ref_for_bq
from techlandscape.decorators import monitor, load_or_persist


@monitor
def count_expansion_level(client, table_ref, data_path):
    """
    Return the number of patents in each expansion_level (e.g: SEED, ANTISEED-XX, L1-XX, etc)
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param data_path: str
    :return: pd.DataFrame
    """
    fio = f"{data_path}{table_ref.table_id}_patent_expansion_level.csv"

    @load_or_persist(fio=fio)
    def main(bq_client, bq_table_ref):
        query = f"""
        SELECT
          expansion_level,
          COUNT(expansion_level) as nb_patent
        FROM
          {format_table_ref_for_bq(bq_table_ref)}
        GROUP BY
          expansion_level
        """
        return bq_client.query(query).to_dataframe()

    df = main(bq_client=client, bq_table_ref=table_ref)
    return df

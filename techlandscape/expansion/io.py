import os

import pandas as pd

from techlandscape.decorators import monitor, timer
from techlandscape.utils import format_table_ref_for_bq, country_clause_for_bq


@monitor
def load_to_bq(f, client, table_ref, job_config):
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
        df = pd.read_csv(f)
    else:
        assert ("publication_number" in f.columns) or (
            f.index.name == "publication_number"
        )
        df = f
    if "expansion_level" in df.columns:
        pass
    else:
        df["expansion_level"] = "SEED"
    client.load_table_from_dataframe(
        df, table_ref, job_config=job_config
    ).result()


@timer
@monitor
def get_expansion_result(flavor, client, table_ref, countries=None):
    """
    Return the result of the expansion.
    E.g: publication_number | expansion_level | abstract
    :param flavor:str, ['*expansion', '*seed']
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pandas.core.frame.DataFrame
    """
    assert flavor in ["*expansion", "*seed"]
    if flavor == "*seed":
        expansion_level_clause = ""
        country_clause = ""
    else:
        expansion_level_clause = "NOT"
        country_clause = (
            f"AND r.country in ({country_clause_for_bq(countries)})"
            if countries
            else ""
        )
    query = f"""
    SELECT
      tmp.publication_number,
      tmp.expansion_level,
      abstract
    FROM
      `patents-public-data.google_patents_research.publications` as r,
      {format_table_ref_for_bq(table_ref)} as tmp
    WHERE
      r.publication_number=tmp.publication_number
      {country_clause}
      AND abstract is not NULL
      AND abstract!=''
      AND expansion_level {expansion_level_clause} LIKE "%SEED%"
    GROUP BY
      publication_number, expansion_level, abstract  
    """
    # nb: we could start by only loading the seed and anti seed to start the classification exercise
    # asap and load the rest of the dataset in the meantime
    # TODO add attributes country_code, assignee.name, assignee.country_code, inventor.name,
    #   inventor.country_code, publication_date, cpc, etc. Later location (from Gaëtan)
    #   -> for later, only on the class 0. NB:
    return client.query(query).to_dataframe()

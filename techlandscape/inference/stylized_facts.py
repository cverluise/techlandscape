from techlandscape.decorators import monitor, load_or_persist
from techlandscape.utils import format_table_ref_for_bq


@monitor
def get_patent_country_date(client, table_ref, data_path):
    """
    Return the patent office and publication_date of patents in <table_ref>
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param data_path: str
    :return: pd.DataFrame
    """
    fio = f"{data_path}{table_ref.table_id}_country_date.csv"

    @load_or_persist(fio=fio)
    def main(bq_client, bq_table_ref):
        query = f"""
        SELECT
          h.publication_number,
          h.expansion_level,
          p.country_code as country_code,
          p.publication_date
        FROM
          `patents-public-data.patents.publications` AS p,
          {format_table_ref_for_bq(bq_table_ref)} AS h
        WHERE
          p.publication_number=h.publication_number
        GROUP BY
          publication_number, p.publication_date, p.country_code, h.expansion_level 
        """
        return bq_client.query(query).to_dataframe()

    df = main(bq_client=client, bq_table_ref=table_ref)
    return df


@monitor
def get_patent_entity(flavor, client, table_ref, data_path):
    """
    Return the "harmonized name" and country of <entity> of patents in <table_ref>
    :param flavor: str, ["inventor", "assignee"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param data_path: str
    :return: pd.DataFrame
    """
    assert flavor in ["inventor", "assignee"]
    fio = f"{data_path}{table_ref.table_id}_patent_{flavor}.csv"

    @load_or_persist(fio=fio)
    def main(e_flavor, bq_client, bq_table_ref):
        query = f"""
        SELECT
          h.publication_number,
          {e_flavor}.name AS {e_flavor}_name,
          {e_flavor}.country_code AS {e_flavor}_country
        FROM
          `patents-public-data.patents.publications` AS p,
          {format_table_ref_for_bq(bq_table_ref)} AS h,
          UNNEST({e_flavor}_harmonized) AS {e_flavor}
        WHERE
          p.publication_number=h.publication_number
        GROUP BY
          publication_number, {e_flavor}_name, {e_flavor}_country
        """
        return bq_client.query(query).to_dataframe()

    df = main(e_flavor=flavor, bq_client=client, bq_table_ref=table_ref)
    return df


@monitor
def get_patent_geoloc(flavor, client, table_ref, data_path):
    """
    Return the geolocation of patent applicants/inventors (<flavor>) of patents in <table_ref>
    :param flavor: str, in ["app", "inv"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :param data_path: str
    :return: pd.DataFrame
    """
    assert flavor in ["app", "inv"]
    fio = f"{data_path}{table_ref.table_id}_patent_{flavor}_geoloc.csv"

    @load_or_persist(fio=fio)
    def main(e_flavor, bq_client, bq_table_ref):
        query = f"""
        SELECT
          seg.publication_number,
          expansion_level,
          appln_id,
          patent_office,
          city,
          lat,
          lng
        from 
          {format_table_ref_for_bq(bq_table_ref)} as seg,
          `brv-patent.external.{e_flavor}_geo_pubnum` as geo
        WHERE
          geo.publication_number=seg.publication_number
        """
        return bq_client.query(query).to_dataframe()

    df = main(e_flavor=flavor, bq_client=client, bq_table_ref=table_ref)
    return df

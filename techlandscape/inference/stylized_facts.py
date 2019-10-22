from techlandscape.decorators import monitor
from techlandscape.utils import format_table_ref_for_bq


@monitor
def get_patent_country_date(client, table_ref):
    """

    :param client:
    :param table_ref:
    :return:
    """
    query = f"""
    SELECT
      h.publication_number,
      h.expansion_level,
      p.country_code as country_code,
      p.publication_date
    FROM
      `patents-public-data.patents.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS h
    WHERE
      p.publication_number=h.publication_number
    GROUP BY
      publication_number, p.publication_date, p.country_code, h.expansion_level 
    """
    return client.query(query).to_dataframe()


@monitor
def get_patent_entity(flavor, client, table_ref):
    """
    Return the "harmonized name" and country of <entity> for patents in <table_ref>
    :param flavor: str, ["inventor", "assignee"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pd.DataFrame
    """
    query = f"""
    SELECT
      h.publication_number,
      {flavor}.name AS {flavor}_name,
      {flavor}.country_code AS {flavor}_country
    FROM
      `patents-public-data.patents.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS h,
      UNNEST({flavor}_harmonized) AS {flavor}
    WHERE
      p.publication_number=h.publication_number
    GROUP BY
      publication_number, {flavor}_name, {flavor}_country
    """
    return client.query(query).to_dataframe()


def get_patent_geoloc(flavor, client, table_ref):
    """
    Return the geolocation of patent applicants/inventors (<flavor>) of patents in <table_ref>
    :param flavor: str, in ["app", "inv"]
    :param client: google.cloud.bigquery.client.Client
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: pd.DataFrame
    """
    assert flavor in ["app", "inv"]
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
      {format_table_ref_for_bq(table_ref)} as seg,
      `brv-patent.external.{flavor}_geo_pubnum` as geo
    WHERE
      geo.publication_number=seg.publication_number
    """
    return client.query(query).to_dataframe()

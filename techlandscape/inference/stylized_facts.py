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
def get_patent_inventor(client, table_ref):
    """

    :param client:
    :param table_ref:
    :return:
    """
    query = f"""
    SELECT
      h.publication_number,
      inventor.name AS inventor_name,
      inventor.country_code AS inventor_country
    FROM
      `patents-public-data.patents.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS h,
      UNNEST(inventor_harmonized) AS inventor
    WHERE
      p.publication_number=h.publication_number
    GROUP BY
      publication_number, inventor_name, inventor_country
    """
    return client.query(query).to_dataframe()


@monitor
def get_patent_assignee(client, table_ref):
    """

    :param client:
    :param table_ref:
    :return:
    """
    query = f"""
    SELECT
      h.publication_number,
      assignee.name AS assignee_name,
      assignee.country_code AS assignee_country
    FROM
      `patents-public-data.patents.publications` AS p,
      {format_table_ref_for_bq(table_ref)} AS h,
      UNNEST(assignee_harmonized) AS assignee
    WHERE
      p.publication_number=h.publication_number
    GROUP BY
      publication_number, assignee_name, assignee_country
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

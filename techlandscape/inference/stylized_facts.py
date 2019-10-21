from techlandscape.decorators import monitor
from techlandscape.utils import format_table_ref_for_bq


# TODO: add location data from Gaëtan
# snippet to get the application_id
# SELECT
#   SPLIT(publication_number, '-')[
# OFFSET
#   (1)] as application_id
# FROM
#   `brv-patent.tech_landscape.hair_dryer_segment`
# TODO: make proper functions


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


# WITH
#   tmp AS (
#   SELECT
#     SPLIT(publication_number, '-')[
#   OFFSET
#     (1)] AS application_id
#   FROM
#     `brv-patent.tech_landscape.hair_dryer_segment`)
# SELECT
#   lat,
#   lng,
#   appln_id
# FROM
#   `patstat2016a.raw.app_geo` AS app_geo
# JOIN
#   tmp
# WHERE
#   app_geo.appln_id=tmp.application_id

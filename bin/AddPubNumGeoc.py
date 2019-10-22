from techlandscape.config import Config
from techlandscape.utils import format_table_ref_for_bq


def create_app2pub(dataset_dest="external"):
    """
    Create an equivalence table between the publication_number and the appln_id located in
    <project_id>.<dataset_dest>.app2pub
    :return:
    """
    config = Config(dataset_id=dataset_dest)
    client = config.client()
    assert dataset_dest in [e.dataset_id for e in list(client.list_datasets())]
    table_ref = config.table_ref("tls211")
    job_config = config.query_job_config(table_id="app2pub", client=client)
    job_config.write_disposition = "WRITE_TRUNCATE"
    query = f"""
    SELECT
      REPLACE(CONCAT(publn_auth, "-", publn_nr, "-", publn_kind), " ", "") AS publication_number,
      appln_id
    FROM
      {format_table_ref_for_bq(table_ref)}
    WHERE
      publn_nr IS NOT NULL
      AND publn_nr != ""
    """
    client.query(query, job_config=job_config)


def add_pubnum_geoc(flavor, dataset_dest="external"):
    """
    Add a publication_number field in the geo table à la de Rassenfosse, Kozak and Seliger
    :param flavor: str, in ["app_geo", "inv_geo"]
    :param dataset_dest: str, name of the destination dataset
    :return:
    """
    config = Config(dataset_id=dataset_dest)
    client = config.client()
    assert dataset_dest in [e.dataset_id for e in list(client.list_datasets())]
    assert flavor in ["app_geo", "inv_geo"]
    # TODO assert app2pub
    table_ref = config.table_ref(flavor, client)
    job_config = config.query_job_config(table_id=flavor, client=client)
    job_config.write_disposition = "WRITE_EMPTY"
    query = f"""
        SELECT
          {flavor}.appln_id,
          app2pub.publication_number,
          patent_office,
          priority_date,
          city,
          lat,
          lng,
          name_0,
          name_1,
          name_2,
          name_3,
          name_4,
          name_5
        FROM
          {format_table_ref_for_bq(table_ref)} AS {flavor},
          `brv-patent.external.app2pub` AS app2pub
        WHERE
          app2pub.appln_id={flavor}.appln_id
    """
    client.query(query, job_config=job_config)


# create_app2pub()
# add_pubnum_geoc("inv_geo_pubnum")
# add_pubnum_geoc("app_geo_pubnum")

# TODO make it executable

import click
from google.cloud import bigquery as bq

from techlandscape.config import Config

"""
    Script to load geo data to bq (step 2):
    1. Load data to storage
        $gsutil -m cp "tls211*.txt" gs://tech_landscape/geocoding/
        $gsutil -m cp "pf_*_external_users.csv" gs://tech_landscape/geocoding/
    Warning, current data might not be up to date
    2. Load data from storage to bq
        $python bin/LoadGeocToBQ.py --flavor "app_geo" --uris 
        "gs://tech_landscape/geocoding/pf_app_external_users.csv"                                                                                             ~/Documents/GitHub/TechLandscape
        $python bin/LoadGeocToBQ.py --flavor "inv_geo" --uris 
        "gs://tech_landscape/geocoding/pf_inv_external_users.csv"                                                                                             ~/Documents/GitHub/TechLandscape
        $python bin/LoadGeocToBQ.py --flavor "tls211" --uris 
        "gs://tech_landscape/geocoding/tls211*.txt"
    3. Create appln_nbr to publn_nbr table
    "SELECT
      REPLACE(CONCAT(publn_auth, "-", publn_nr, "-", publn_kind), " ", "") AS publication_number,
      appln_id
    FROM
      `brv-patent.external.tls211`
    WHERE
      publn_nr IS NOT NULL
      AND publn_nr != ""
  "
"""
# TODO update geoc data

config = Config(dataset_id="external")
client = config.client()

INT = "INTEGER"
STR = "STRING"
BOO = "BOOLEAN"
FLO = "FLOAT"
DAT = "DATE"
GEO = "GEOGRAPHY"

NULL = "NULLABLE"
REQ = "REQUIRED"


class Schema:
    """
    Schema of App & Inv geoloc, from
    """

    def __init__(self):
        self.tls211 = [
            bq.SchemaField("pat_publn_id", INT, REQ, None, ()),
            bq.SchemaField("publn_auth", STR, NULL, None, ()),
            bq.SchemaField("publn_nr", STR, NULL, None, ()),
            bq.SchemaField("publn_nr_original", STR, NULL, None, ()),
            bq.SchemaField("publn_kind", STR, NULL, None, ()),
            bq.SchemaField("appln_id", INT, REQ, None, ()),
            bq.SchemaField("publn_date", DAT, NULL, None, ()),
            bq.SchemaField("publn_lg", STR, NULL, None, ()),
            bq.SchemaField("publn_first_grant", INT, NULL, None, ()),
            bq.SchemaField("publn_claims", INT, NULL, None, ()),
        ]
        self.inv_geo = [
            bq.SchemaField("appln_id", INT, REQ, None, ()),
            bq.SchemaField("patent_office", STR, NULL, None, ()),
            bq.SchemaField("priority_date", DAT, NULL, None, ()),
            bq.SchemaField("name_0", STR, NULL, None, ()),
            bq.SchemaField("name_1", STR, NULL, None, ()),
            bq.SchemaField("name_2", STR, NULL, None, ()),
            bq.SchemaField("name_3", STR, NULL, None, ()),
            bq.SchemaField("name_4", STR, NULL, None, ()),
            bq.SchemaField("name_5", STR, NULL, None, ()),
            bq.SchemaField("city", STR, NULL, None, ()),
            bq.SchemaField("lat", FLO, NULL, None, ()),
            bq.SchemaField("lng", FLO, NULL, None, ()),
            bq.SchemaField("data_source", STR, NULL, None, ()),
            bq.SchemaField("coord_source", STR, NULL, None, ()),
            bq.SchemaField("source", INT, NULL, None, ()),
            bq.SchemaField("priority_year", INT, NULL, None, ()),
        ]
        self.app_geo = [
            bq.SchemaField("appln_id", INT, REQ, None, ()),
            bq.SchemaField("patent_office", STR, NULL, None, ()),
            bq.SchemaField("priority_date", DAT, NULL, None, ()),
            bq.SchemaField("name_0", STR, NULL, None, ()),
            bq.SchemaField("name_1", STR, NULL, None, ()),
            bq.SchemaField("name_2", STR, NULL, None, ()),
            bq.SchemaField("name_3", STR, NULL, None, ()),
            bq.SchemaField("name_4", STR, NULL, None, ()),
            bq.SchemaField("name_5", STR, NULL, None, ()),
            bq.SchemaField("city", STR, NULL, None, ()),
            bq.SchemaField("lat", FLO, NULL, None, ()),
            bq.SchemaField("lng", FLO, NULL, None, ()),
            bq.SchemaField("data_source", STR, NULL, None, ()),
            bq.SchemaField("coord_source", STR, NULL, None, ()),
            bq.SchemaField("source", INT, NULL, None, ()),
            bq.SchemaField("priority_year", INT, NULL, None, ()),
        ]


def load_gs_to_bq(flavor, uris):
    """

    :param flavor:
    :param uris:
    :return:
    """
    assert flavor in ["app_geo", "inv_geo", "tls211"]
    table_ref = config.table_ref(flavor, client=client)
    job_config = config.load_job_config()
    job_config.skip_leading_rows = 1
    job_config.max_bad_records = 10
    job_config.source_format = bq.SourceFormat.CSV
    if flavor == "app_geo":
        job_config.schema = Schema().app_geo
    elif flavor == "inv_geo":
        job_config.schema = Schema().inv_geo
    else:
        job_config.schema = Schema().tls211

    load_job = client.load_table_from_uri(
        source_uris=uris, destination=table_ref, job_config=job_config
    )
    load_job.result()
    assert load_job.state == "DONE"


if __name__ == "__main__":

    @click.command()
    @click.option(
        "--flavor",
        help="Kind of table to be loaded. Currently supported 'app_geo', 'inv_geo' and "
        "'tls211'",
    )
    @click.option("--uris", help="GS uris. Nb: Wildcard * allowed")
    def main(flavor, uris):
        load_gs_to_bq(flavor=flavor, uris=uris)

    main()

# flavor = "tls211"
# table_ref = config.table_ref(flavor, client=client)
# job_config = config.load_job_config()
# job_config.write_disposition = "WRITE_TRUNCATE"
# job_config.skip_leading_rows = 1
# job_config.max_bad_records = 10
# job_config.source_format = bq.SourceFormat.CSV
# job_config.schema = Schema().tls211
# uri = 'gs://tech_landscape/geocoding/tls211_part*.txt'
#
# load_job = client.load_table_from_uri(
#     source_uris=uri,
#     destination=table_ref,
#     job_config=job_config)
# load_job.result()
# assert load_job.state == 'DONE'

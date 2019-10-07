import json
from google.cloud import bigquery as bq

english_speaking_offices = [
    "AP",
    "AU",
    "CA",
    "EA",
    "EP",
    "GB",
    "HK",
    "IE",
    "IN",
    "NZ",
    "SG",
    "US",
    "ZA",
]


def format_table_ref_for_bq(table_ref):
    """
    Return the table_ref as string formated for bq queries. E.g `brv-patent.tech_landscape.tmp`
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: str
    """
    return f"`{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"


def country_clause_for_bq():
    """
    Return the list of countries of interest in a bq understandable way
    Resource: https://worldwide.espacenet.com/help?locale=en_EP&method=handleHelpTopic&topic
    =countrycodes
    :return: str '"AP","AU","CA",...'
    """
    return ",".join(
        list(map(lambda x: '"' + str(x) + '"', english_speaking_offices))
    )


def flatten(l):
    """
    Return a list of list as a flat list. E.g [[],[],...] -> [,,...]
    :param l: List[list]
    :return: list
    """
    return [item for sublist in l for item in sublist if item]


class Config:
    def __init__(self):
        self.config_dict = json.load(open("config.json", "rb"))
        self.project_id = self.config_dict["project_id"]
        self.dataset_id = self.config_dict["dataset_id"]

    def client(self):
        """
        :return:
        """
        return bq.Client(project=self.project_id)

    def table_ref(self, table_id, client=None):
        """
        :param client: bq.Client or None, if None, populated with config.json attr
        :param table_id: str
        :return: table_ref
        """
        if not client:
            client = self.client()
        return client.dataset(dataset_id=self.dataset_id).table(
            table_id=table_id
        )

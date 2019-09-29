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

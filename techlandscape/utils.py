import json
import random
import string

from wasabi import Printer

from techlandscape.decorators import timer

country_groups = {
    "english_speaking": [
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
    ],
    "g7": ["CA", "FR", "DE", "IT", "JP", "GB", "US"],
    "brics": ["BR", "RU", "IN", "CN", "ZA"],
}


def iso2cnt(ent_list, flavor):
    """
    Return the list of the full country names (resp the iso2) matching the list of countries (
    resp iso2) in <ent_list>. The <flavor> param refers to the flavor of entities in <ent_list>.
    :param ent_list: iter
    :param flavor: str, ["iso", "cnt"]
    :return:
    """
    assert flavor in ["iso", "cnt"]
    tmp = json.load(open("data/lib/iso2cnt.json", "r"))
    if flavor == "iso":
        res = [v for k, v in tmp.items() if k in ent_list]
    else:
        res = [k for k, v in tmp.items() if v in ent_list]
    nb_missing = len(ent_list) - len(res)
    if nb_missing > 0:
        msg = Printer()
        msg.warn(f"{nb_missing} entity.ies could not be found.")
        if nb_missing == len(ent_list):
            msg.info(f"You might be using the wrong 'flavor'.")
    return res


def format_table_ref_for_bq(table_ref):
    """
    Return the table_ref as string formated for bq queries. E.g `brv-patent.tech_landscape.tmp`
    :param table_ref: google.cloud.bigquery.table.TableReference
    :return: str
    """
    return f"`{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"


def country_clause_for_bq(iso_list, to_cnt=True):
    """
    Return the list of countries of interest in a bq understandable way
    Resource: https://worldwide.espacenet.com/help?locale=en_EP&method=handleHelpTopic&topic
    =countrycodes
    :param iso_list: list, list of iso2 countries
    :param to_cnt: bool, True if conversion to cnt needes (E.g. if applied to
    google_patents_research dataset)
    :return: str '"AP","AU","CA",...'
    """
    if to_cnt:
        ent_list = iso2cnt(iso_list, "iso")
    else:
        ent_list = iso_list
    return ",".join(list(map(lambda x: '"' + str(x) + '"', ent_list)))


def flatten(l):
    """
    Return a list of list as a flat list. E.g [[],[],...] -> [,,...]
    :param l: List[list]
    :return: list
    """
    return [item for sublist in l for item in sublist if item]


@timer
def breathe():
    pass


def get_uid(n=6):
    """
    Generate a random string of letters and digits
    :param n: int, uid length
    :return: str
    """
    seq = string.ascii_letters.lower() + string.digits
    return "".join(random.choice(seq) for i in range(n))

from techlandscape.exception import SmallSeed
from techlandscape.utils import country_string_for_bq


def project_id(key, client):
    """
    Return the project with data required for expansion depending on the key.
    If key is publication_number, the project is patents-public-data, else (family_id) this is the user
    project in which the family level tables were placed
    :param key: str, in ["publication_number", "family_id"]
    :param client: bq.Client
    :return: str
    """
    assert key in ["publication_number", "family_id"]
    return (
        "patents-public-data"
        if key == "publication_number"
        else client.project
    )


def country_prefix(key):
    """
    Return a prefix to unnest country field for tables at the family level (empty else)
    :param key: str, in ["publication_number", "family_id"]
    :return: str
    """
    assert key in ["publication_number", "family_id"]
    return (
        "" if key == "publication_number" else ", UNNEST(country) as country"
    )


def country_clause(countries):
    """
    Return a restrictive clause on countries in countries not None
    :param countries: List[str], ISO2 countries we are interested in
    :return: str
    """
    return (
        f"AND country in ({country_string_for_bq(countries)})"
        if countries
        else ""
    )


def pc_like_clause(flavor, pc_list, sub_group=False):
    """
    Return a close to restrict to pc.code which contain at least one of the pc codes in pc_list
    :param flavor: str, in ["cpc", "ipc"]
    :param pc_list: List[str]
    :param sub_group: bool, True if sub-group, False if group level
    :return: str
    """
    assert flavor in ["cpc", "ipc"]
    pc_list_ = (
        list(map(lambda x: x.split("/")[0], pc_list)) if sub_group else pc_list
    )
    return (
        "("
        + " OR ".join(
            set(
                list(
                    map(
                        lambda x: f'{flavor}.code LIKE "%' + x + '%"', pc_list_
                    )
                )
            )
        )
        + ")"
    )


def get_antiseed_size(seed_size, min_size=100, threshold_size=250):
    """
    Return the antiseed size such that the seed never represents less than 10% of the sample
    between <min_size> and <threshold_size> (linear) and 20% of the sample above <threshold_size>
    (constant). Seeds with less than <min_size> patents raise an error.
    :param seed_size: int
    :param min_size: int
    :param threshold_size: int
    :return:
    """
    if seed_size < min_size:
        raise SmallSeed("Danger Zone: your seed is too small. Don't cross!")
    elif min_size <= seed_size < threshold_size:
        share_seed = (
            0.1 + (seed_size - min_size) / (threshold_size - min_size) * 0.1
        )
        antiseed_size = int(seed_size / share_seed)
    else:
        share_seed = 0.2
        antiseed_size = int(seed_size / share_seed)
    return antiseed_size

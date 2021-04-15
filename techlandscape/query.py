from techlandscape.exception import SmallSeed
from techlandscape.utils import get_country_string_bq
from typing import List
from google.cloud import bigquery


def get_project_id(key: str, client: bigquery.client.Client) -> str:
    """
    Return the name of the project used for expansion depending on the key ("publication_number" or "family_id").
    If `key` is "publication_number", the project is patents-public-data, else (`family_id`) this is the `client`'s
    project
    """
    assert key in ["publication_number", "family_id"]
    return (
        "patents-public-data"
        if key == "publication_number"
        else client.project
    )


def get_country_prefix(key: str) -> str:
    """
    Return a prefix to unnest country field for tables at the family level (else empty)
    """
    assert key in ["publication_number", "family_id"]
    return (
        "" if key == "publication_number" else ", UNNEST(country) as country"
    )


def get_country_clause(countries: List[str]) -> str:
    """
    Return a restrictive clause on countries if `countries` not None
    """
    return (
        f"AND country in ({get_country_string_bq(countries)})"
        if countries
        else ""
    )


def get_pc_like_clause(
    flavor: str, pc_list: List[str], sub_group: bool = False
):
    """
    Return a clause to restrict to cpc/ipc.code which contain at least one of the pc codes in `pc_list`
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


def get_antiseed_size(
    seed_size: int, min_size: int = 100, threshold_size: int = 250
) -> int:
    """
    Return the antiseed size such that the seed never represents less than 10% of the sample
    between `min_size` and `threshold_size` (linear) and 20% of the sample above `threshold_size`
    (constant). Seeds with less than `min_size` patents raise an error.
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

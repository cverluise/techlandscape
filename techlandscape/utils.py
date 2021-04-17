import random
import string
import json
import typer
from pathlib import Path
from typing import List
from google.cloud import bigquery
from google.oauth2 import service_account
from techlandscape.lib import ISO2CNT
from techlandscape.decorators import timer
from techlandscape.exception import SmallSeed

ok = "\u2713"
not_ok = "\u2717"


def get_bq_client(credentials: Path):
    """"""
    credentials = service_account.Credentials.from_service_account_file(
        credentials, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = bigquery.Client(
        credentials=credentials, project=credentials.project_id
    )
    return client


def get_bq_job_done(
    query: str,
    destination_table: str,
    credentials: Path,
    write_disposition: str = "WRITE_TRUNCATE",
    verbose: bool = True,
    **kwargs,
):
    """Config and execute query job.
    Nb: `**kwargs` passed to bigquery.QueryJobConfig"""
    job_config = bigquery.QueryJobConfig(
        destination=destination_table,
        write_disposition=write_disposition,
        **kwargs,
    )
    client = get_bq_client(credentials)
    if verbose:
        typer.secho(f"Start:\n{query}", fg=typer.colors.BLUE)
    client.query(query, job_config=job_config).result()

    typer.secho(
        f"{ok}Query results saved to {destination_table}",
        fg=typer.colors.GREEN,
    )


def iso2cnt(ent_list: List[str], flavor: str) -> List[str]:
    """Return the list of country full names (resp the iso2) matching the list of countries (resp iso2) in `ent_list`.
    The `flavor` param refers to the flavor of entities in `ent_list`.
    """
    assert flavor in ["iso", "cnt"]
    if flavor == "iso":
        res = [v for k, v in ISO2CNT.items() if k in ent_list]
    else:
        res = [k for k, v in ISO2CNT.items() if v in ent_list]
    nb_missing = len(ent_list) - len(res)
    if nb_missing > 0:
        typer.secho(
            f"Unable to find {nb_missing} entity.ies.",
            color=typer.colors.YELLOW,
        )
        if nb_missing == len(ent_list):
            typer.secho(
                f"{not_ok}Unable to find entity.ies. You might be using the wrong 'flavor'",
                color=typer.colors.RED,
            )
    return res


def format_table_ref_for_bq(table_ref: bigquery.table.TableReference) -> str:
    """
    Return the table_ref as string formated for bq queries. E.g `brv-patent.tech_landscape.tmp`
    """
    # TODO deprecate
    return f"`{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"


def get_country_string_bq(iso_list: List[str], to_cnt: bool = True) -> str:
    """
    Return the list of countries of interest in a bq compatible format (e.g. '"AP","AU","CA",...')
    Resource: https://worldwide.espacenet.com/help?locale=en_EP&method=handleHelpTopic&topic
    =countrycodes
    """
    if to_cnt:
        ent_list = iso2cnt(iso_list, "iso")
    else:
        ent_list = iso_list
    return ",".join(list(map(lambda x: '"' + str(x) + '"', ent_list)))


def flatten(l: List[List]) -> List:
    """
    Return a list of list as a flat list. E.g [[],[],...] -> [,,...]
    """
    return [item for sublist in l for item in sublist if item]


@timer
def breathe():
    pass


def get_uid(n: int = 6) -> str:
    """
    Generate a random string of letters and digits of length `n`
    """
    seq = string.ascii_letters.lower() + string.digits
    return "".join(random.choice(seq) for i in range(n))


def get_project_id(key: str, credentials: Path) -> str:
    """
    Return the name of the project used for expansion depending on the key ("publication_number" or "family_id").
    If `key` is "publication_number", the project is patents-public-data, else (`family_id`) this is the `client`'s
    project
    """
    assert key in ["publication_number", "family_id"]

    return (
        "patents-public-data"
        if key == "publication_number"
        else json.loads(Path(credentials).open("r").read()).get("project_id")
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
    Return a close to restrict to pc.code which contain at least one of the pc codes in pc_list
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

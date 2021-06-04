import random
import string
import json
import typer
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from techlandscape.lib import ISO2CNT
from techlandscape.decorators import timer
from techlandscape.exception import SmallSeed, SMALL_SEED_MSG
from techlandscape.enumerators import PrimaryKey, TechClass
from smart_open import open

app = typer.Typer()

ok = "\u2713"
not_ok = "\u2717"


def get_bq_client(credentials: Path) -> bigquery.Client:
    """"""
    credentials = service_account.Credentials.from_service_account_file(
        credentials, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
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
        destination=destination_table, write_disposition=write_disposition, **kwargs
    )
    client = get_bq_client(credentials)
    if verbose:
        typer.secho(f"Start:\n{query}", fg=typer.colors.BLUE)
    client.query(query, job_config=job_config).result()

    typer.secho(
        f"{ok}Query results saved to {destination_table}", fg=typer.colors.GREEN
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
            f"Unable to find {nb_missing} entity.ies.", color=typer.colors.YELLOW
        )
        if nb_missing == len(ent_list):
            typer.secho(
                f"{not_ok}Unable to find entity.ies. You might be using the wrong 'flavor'",
                color=typer.colors.RED,
            )
    return res


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


def get_project_id(primary_key: PrimaryKey, credentials: Path) -> str:
    """
    Return the name of the project used for expansion depending on the primary_key ("publication_number" or "family_id").
    If `primary_key` is "publication_number", the project is patents-public-data, else (`family_id`) this is the `client`'s
    project
    """

    return (
        "patents-public-data"
        if primary_key.value == PrimaryKey.publication_number.value
        else json.loads(Path(credentials).open("r").read()).get("project_id")
    )


def get_country_prefix(primary_key: PrimaryKey) -> str:
    """
    Return a prefix to unnest country field for tables at the family level (else empty)
    """

    return (
        ""
        if primary_key.value == PrimaryKey.publication_number.value
        else ", UNNEST(country_code) as country_code"
    )


def get_country_clause(countries: List[str]) -> str:
    """
    Return a restrictive clause on countries if `countries` not None
    """
    return f"AND country in ({get_country_string_bq(countries)})" if countries else ""


def get_pc_like_clause(
    tech_class: TechClass, pc_list: List[str], sub_group: bool = False
):
    """
    Return a close to restrict to pc.code which contain at least one of the pc codes in pc_list
    """
    pc_list_ = list(map(lambda x: x.split("/")[0], pc_list)) if sub_group else pc_list
    return (
        "("
        + " OR ".join(
            set(
                list(
                    map(
                        lambda x: f'{tech_class.value}.code LIKE "%' + x + '%"',
                        pc_list_,
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
        share_seed = 0.1 + (seed_size - min_size) / (threshold_size - min_size) * 0.1
        antiseed_size = int(seed_size / share_seed)
    else:
        share_seed = 0.2
        antiseed_size = int(seed_size / share_seed)
    return antiseed_size


def densify_var(
    df: pd.DataFrame, group: str, func: callable, var: str, n_keep: int
) -> pd.DataFrame:
    """
    Return `df` with a denser `group`; only the `n_keep` (non null) categories
    wrt `var` are kept, other categories of `group` are pulled into "others"
    """
    tmp = df.copy()
    var_clause = tmp.groupby(group).apply(func)[var].nlargest(n_keep + 1).index
    # filter out "" and None
    var_clause = list(filter(lambda x: x, var_clause))[:n_keep]
    dense_var = list(
        map(lambda x: x if x in var_clause else "others", tmp[group].values)
    )
    tmp[group] = dense_var
    return tmp


def get_share_test(
    seed_size: int, min_size: int = 100, threshold_size: int = 250
) -> float:
    """
    Return the antiseed size such that the seed never represents less than 10% of the sample
    between `min_size` and `threshold_size` (linear) and 20% of the sample above `threshold_size`
    (constant). Nb: seeds with less than `min_size` patents raise an error.
    """
    if seed_size < min_size:
        raise SmallSeed(SMALL_SEED_MSG)
    elif min_size <= seed_size < threshold_size:
        share_test = 0.5 - (seed_size - min_size) / (threshold_size - min_size) * 0.3
    else:
        share_test = 0.2
    return share_test


def get_config(config: Path) -> dict:
    """Return config from config file"""
    return yaml.load(Path(config).open("r"), Loader=yaml.FullLoader)


def get_train_test(
    classif_df: pd.DataFrame, random_state: str = 42
) -> Tuple[Any, Any, Any, Any]:
    """Return arrays of train, test data (features and output)"""

    def get_expansion_level_to_label(x: str) -> int:
        """Turn the expansion level into a binary var with value 0 if 'SEED' else 0"""
        return 0 if x == "SEED" else 1

    def get_test_ratio(
        seed_size: int, min_size: int = 100, threshold_size: int = 250
    ) -> float:
        """
        Return the antiseed size such that the seed never represents less than 10% of the sample between `min_size` and
        `threshold_size` (linear) and 20% of the sample above `threshold_size` (constant). Seeds with less than `min_size`
        patents raise an error.
        """
        if seed_size < min_size:
            raise SmallSeed("Danger Zone: your seed is too small. Don't cross!")
        elif min_size <= seed_size < threshold_size:
            test_ratio_ = (
                0.5 - (seed_size - min_size) / (threshold_size - min_size) * 0.3
            )
        else:
            test_ratio_ = 0.2
        return test_ratio_

    var_required = ["abstract", "expansion_level"]

    for v in var_required:
        assert v in classif_df.columns

    # prepare labels
    classif_df["label"] = classif_df["expansion_level"].apply(
        lambda x: get_expansion_level_to_label(x)
    )
    # shuffle
    classif_df.sample(frac=1, random_state=random_state)

    # train test set
    X = classif_df["abstract"].to_list()
    y = classif_df["label"].to_list()
    test_size = get_test_ratio(len(classif_df.query("expansion_level=='SEED'")))
    texts_train, texts_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return texts_train, texts_test, y_train, y_test


@app.command()
def correct_annotations(buggy_file: str, corr_file: str):
    """
    Print `buggy_file` with corrections from `corr_file` (using `"family_id"` key) to stdout

    Arguments:
         buggy_file: buggy file path
         corr_file: correction file path

    **Usage:**
    ```python
    techlandscape utils correct-annotations <buggy-file> <corr-file>
    ```
    """

    def get_corr_index(corr_file):
        corr_index = {}
        with open(corr_file, "r") as lines:
            for line in lines:
                line = json.loads(line)
                corr_index.update({line["family_id"]: line})
        return corr_index

    def update_corrected_lines(buggy_file, corr_index):
        with open(buggy_file, "r") as lines:
            for line in lines:
                line = json.loads(line)
                line.update(corr_index.get(line["family_id"], line))
                typer.echo(json.dumps(line))

    corr_index = get_corr_index(corr_file)
    update_corrected_lines(buggy_file, corr_index)


@app.command()
def add_practical_fields(file):
    """Print `file` with human readable `"accept"` and `"expansion_level"` fields to stdout

    Arguments:
        file: file path

    **Usage:**
        ```shell
        techlandscape utils add-practical-fields <file>
        ```
    """
    with open(file, "r") as lines:
        for line in lines:
            line = json.loads(line)
            # human readable accept
            accepts_ = line["accept"]
            options_ = [option["text"] for option in line["options"]]
            line.update({"accept_text": [options_[accept_] for accept_ in accepts_]})
            # expansion_level
            answer_ = line.get("answer")
            expansion_level_ = (
                "SEED"
                if answer_ == "accept"
                else ("ANTISEED-manual" if answer_ == "reject" else None)
            )
            line.update({"expansion_level": expansion_level_})
            typer.echo(json.dumps(line))


@app.command()
def flatten_nested_vars(file: Path, nested_vars: str):
    """Print `file` with flattened `nested_vars` to stdout

    Arguments:
        file: file path
        nested_vars: comma-separated list of nested variables to be flattened

    !!! info
        This is useful to enable BQ absorption of nested fields as NULLABLE fields (rather than REPEATED).
    """

    def flatten_nested_field(nested_field: dict) -> List[dict]:
        """Return a REPEATED field as a NULLABLE field

        **Example:**
            ```raw
            I: {"publication_number":["CN-105537038-B","US-9846376-B2"]}
            O: [{"publication_number":"CN-105537038-B"},{"publication_number":"US-9846376-B2"}]
            ```
        """
        out = []
        for key, values in nested_field.items():
            for value in values:
                out += [{key: value}]
        return out

    def flatten_nested_var(nested_var: list) -> list:
        """Return a var with REPEATED nested fields as a var with NULLABLE nested fields

        !!! warning
            only one field supported so far
        """
        if len(nested_var) == 0:
            out = []
        elif len(nested_var) == 1:
            out = flatten_nested_field(nested_var[0])
        else:
            raise ValueError(f"Too many fields (only 1 field supported): {nested_var}")
        return out

    nested_vars = nested_vars.split(",")
    with open(file, "r") as lines:
        for line in lines:
            try:
                line = json.loads(line)
                for nested_var in nested_vars:
                    line.update({nested_var: flatten_nested_var(line.get(nested_var))})
                typer.echo(json.dumps(line))
            except json.decoder.JSONDecodeError as e:
                typer.secho(
                    "\n".join([e.__str__(), line]), err=True, color=typer.colors.YELLOW
                )
                pass


@app.command()
def train_test_split(
    file: Path,
    train: Path,
    test: Path,
    test_ratio: float = 0.2,
    seed_share: float = 1,
    balanced: bool = True,
    random_seed: int = None,
):
    """
    Return the shuffled train/test sets saved to `train`, `test`. If balanced is true, the input data is pre-processed
    to make it balanced (seed=1/3, antiseed_manual=1/3 and antiseed_AF=1/3).

    Arguments:
        file: training data source file path
        train: train set saving file path
        test: test set saving file path
        test_ratio: share of the balanced training set dedicated to the test set
        seed_share: share of the seed to be randomly drawn. nb: other strata of the balanced training set are affected by this choice du to balancing
        balanced: whether the input training data should be balanced or not
        random_seed: random seed

    **Usage:**
        ```shell
        techlandscape utils train-test-split data/training_additivemanufacturing.jsonl train_additivemanufacturing.jsonl test_additivemanufacturing.jsonl
        ```

    """
    random.seed(random_seed)
    with file.open("r") as lines:
        seed, antiseed_manual, antiseed_af = [], [], []
        for line in lines:
            line = json.loads(line)
            expansion_level = line.get("expansion_level")
            if expansion_level == "SEED":
                seed += [line]
            elif expansion_level == "ANTISEED-manual":
                antiseed_manual += [line]
            elif expansion_level == "ANTISEED-AF":
                antiseed_af += [line]
            else:
                typer.secho(f"{line}", err=True, color=typer.colors.YELLOW)
        if balanced:
            size_seed = int(np.ceil(len(seed) * seed_share))
            seed = random.sample(seed, k=size_seed)
            antiseed_manual = random.sample(antiseed_manual, k=size_seed)
            antiseed_af = random.sample(antiseed_af, k=size_seed)
        egs = [*seed, *antiseed_manual, *antiseed_af]
        random.shuffle(egs)

        split = int(np.ceil(len(egs) * test_ratio))
        egs_test, egs_train = egs[:split], egs[split:]
        with test.open("w") as fout:
            for eg in egs_test:
                fout.write(json.dumps(eg) + "\n")
        with train.open("w") as fout:
            for eg in egs_train:
                fout.write(json.dumps(eg) + "\n")


if __name__ == "__main__":
    app()

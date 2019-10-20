import click
import numpy as np
import pandas as pd
import os

from techlandscape.config import Config
from techlandscape.expansion.antiseed import (
    draw_af_antiseed,
    draw_aug_antiseed,
)
from techlandscape.expansion.expansion import (
    get_important_pc,
    pc_expansion,
    citation_expansion,
)
from techlandscape.expansion.io import load_to_bq, get_expansion_result
from techlandscape.expansion.utils import get_antiseed_size
from techlandscape.utils import breathe


def full_expansion(
    seed_file, table_name, data_path, pc_flavor, bq_config=None
):
    # TODO check that the expansion is already persisted

    if os.path.isfile(f"{data_path}{table_name}_classif.csv.gz"):
        classif_df = pd.read_csv(
            data_path + f"{table_name}_classif.csv.gz",
            compression="gzip",
            index_col=0,
        )
    else:
        config = bq_config if bq_config else Config()
        client = config.client()
        table_ref = config.table_ref(table_name, client=client)
        load_job_config = config.load_job_config()
        query_job_config = config.query_job_config(table_name, client=client)

        load_to_bq(seed_file, client, table_ref, load_job_config)

        breathe()
        important_pc = get_important_pc(pc_flavor, 2, client, table_ref)
        important_pc.to_csv(
            data_path + f"{table_name}_important_pc.csv.gz", compression="gzip"
        )
        important_pc_sub = list(
            important_pc.sort_values(by="odds", ascending=False)
            .iloc[:50]
            .index
        )
        pc_expansion(pc_flavor, important_pc_sub, client, query_job_config)

        breathe()
        citation_expansion(
            "citation", "L1", client, table_ref, query_job_config
        )
        citation_expansion(
            "cited_by", "L1", client, table_ref, query_job_config
        )

        breathe()
        citation_expansion(
            "citation", "L2", client, table_ref, query_job_config
        )
        citation_expansion(
            "cited_by", "L2", client, table_ref, query_job_config
        )

        seed_size = len(np.loadtxt(seed_file, dtype="str")) - 1
        antiseed_size = int(get_antiseed_size(seed_size) / 2)
        draw_af_antiseed(antiseed_size, client, table_ref, query_job_config)
        draw_aug_antiseed(
            antiseed_size,
            pc_flavor,
            important_pc_sub,
            client,
            table_ref,
            query_job_config,
        )

        classif_df = get_expansion_result("*seed", client, table_ref)
        classif_df.to_csv(
            data_path + f"{table_name}_classif.csv.gz", compression="gzip"
        )
    return classif_df


if __name__ == "__main__":

    @click.command()
    @click.option(
        "--seed-file",
        help="Seed file. CSV format. Header: 'publication_number', "
        "values: 'CC-NNNN..-SS'",
    )
    @click.option(
        "--table_name", help="Name of the expansion table in BQ. E.g 'LTE'"
    )
    @click.option(
        "--data_path",
        help="Path of the folder for logging intermediary output. E.g "
        "'data/persist/lte'",
    )
    @click.option(
        "--pc_flavor",
        help="Patent class flavor for patent class expansion. Currently "
        "supported: 'ipc', 'cpc'",
    )
    def main(seed_file, table_name, data_path, pc_flavor):
        full_expansion(seed_file, table_name, data_path, pc_flavor)

    main()

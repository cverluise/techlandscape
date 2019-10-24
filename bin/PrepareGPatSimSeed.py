import glob
import os

import click
import pandas as pd

from techlandscape.utils import get_uid


# files = "data/seed/*20191023.csv"


@click.command(
    help="E.g python bin/PrepareGPatSimSeed.py --files 'data/seed/*20191023.csv' > "
    "data/seed/20191024.csv "
)
@click.option("--files", help='"*" wildcard enabled')
@click.option("--seed_size", type=int, default=300, help="Size of the seed")
def main(files, seed_size):
    files = glob.glob(files)
    for f in files:
        root, file = os.path.split(f)
        ext = ".".join(file.split(".")[1:])
        file = file.split("_")[0]

        df = pd.read_csv(f, index_col=0)

        df.index.name = "publication_number"
        df = df.drop(df.columns, axis=1)

        random_df = df.sample(seed_size)
        uid = get_uid()
        file_name = f"{file}_{uid}.{ext}"
        random_df.to_csv(os.path.join(root, file_name))
        print(",".join([file_name, '"random"', str(seed_size)]))

        nfirst_df = df.iloc[:seed_size]
        uid = get_uid()
        file_name = f"{file}_{uid}.{ext}"
        nfirst_df.to_csv(os.path.join(root, f"{file}_{uid}.{ext}"))
        print(",".join([file_name, '"most similar"', str(seed_size)]))


if __name__ == "__main__":
    main()

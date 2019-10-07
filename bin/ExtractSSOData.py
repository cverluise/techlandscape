import os

import click
import pandas as pd
from tqdm import tqdm

from techlandscape.decorators import monitor


# root = "/Volumes/HD_CyrilVerluise/SearleSSODatabase/"


@click.command()
@click.option(
    "--path",
    help="Root folder of the Searle SSO database. E.g.: "
    "/Volumes/.../SearleSSODatabase/ ",
)
@monitor
def main(path):
    for folder in list(filter(lambda x: "." not in x, os.listdir(path))):
        for file in tqdm(
            list(filter(lambda x: ".dta" in x, os.listdir(path + folder)))
        ):
            if os.path.exists(path + folder + f"/{file.split('.')[0]}.csv"):
                pass
            else:
                if file == "SCDB_declared_number.dta":
                    tmp_chunk = pd.read_stata(
                        path + folder + f"/{file}", chunksize=100000
                    )
                    for i, tmp in enumerate(tmp_chunk):
                        tmp.to_csv(
                            path
                            + folder
                            + f"/{file.split('.')[0]}-part_{i}.csv"
                        )
                else:
                    print(folder, file)
                    tmp = pd.read_stata(path + folder + f"/{file}")
                    tmp.to_csv(path + folder + f"/{file.split('.')[0]}.csv")


if __name__ == "__main__":
    main()

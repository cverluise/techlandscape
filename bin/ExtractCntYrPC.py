from techlandscape.config import Config
from techlandscape.decorators import monitor
from wasabi import Printer
import click
import asyncio
import os

config = Config()
client = config.client()
msg = Printer()


async def get_cntyr_pc_count(flavor="cpc", lower_bound=50):
    assert flavor in ["cpc", "ipc"]
    query = f"""
    WITH
      tmp AS (
      SELECT
        {flavor}.code AS {flavor},
        COUNT({flavor}.code) AS count,
        CAST(ROUND(p.publication_date/10000, 0) AS INT64) AS publication_year,
        p.country_code
      FROM
        `patents-public-data.patents.publications` AS p,
        UNNEST({flavor}) AS {flavor}
      GROUP BY
        {flavor},
        publication_year,
        country_code)
    SELECT
      *
    FROM
      tmp
    WHERE
      count > {lower_bound}
    """
    return client.query(query).to_dataframe()


async def get_cntyr_pat_count():
    query = f"""
    SELECT
      COUNT(DISTINCT(publication_number)) AS count,
      country_code,
      CAST(ROUND(publication_date/10000, 0) AS INT64) AS publication_year
    FROM
      `patents-public-data.patents.publications`
    GROUP BY
      country_code,
      publication_year
      """
    return client.query(query).to_dataframe()


async def a_main(path, flavor, lower_bound):
    num_task = asyncio.create_task(get_cntyr_pc_count(flavor=flavor, lower_bound=lower_bound))
    denom_task = asyncio.create_task(get_cntyr_pat_count())

    await num_task
    await denom_task
    num_df = num_task.result()
    denom_df = denom_task.result()

    df = num_df.merge(denom_df, on=["country_code", "publication_year"], how="left", suffixes=("_num", "_denom"))
    df["freq"] = df["count_num"] / df["count_denom"]
    file = os.path.join(path, f"{flavor}_freq_cntyr.csv.gz")
    df[["publication_year", "country_code", f"{flavor}", "freq"]].to_csv(file,
                                                                         compression="gzip")
    msg.info(f"Saved {file}")


@click.command()
@click.option("--path", help="Destination folder. E.g. 'data'")
@click.option("--flavor", default="cpc", help="Patent class, in ['cpc','ipc']")
@click.option("--lower_bound", default=50, type=int, help="Minimum number of occurences in a bucket. 0 below.")
@monitor
def main(path, flavor, lower_bound):  # artefact, click failing on async functions
    asyncio.run(a_main(path, flavor, lower_bound))


if __name__ == "__main__":
    main()

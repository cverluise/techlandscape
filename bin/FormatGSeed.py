import click
import numpy as np
import pandas as pd
from google.cloud import bigquery as bq

from techlandscape.decorators import monitor
from techlandscape.utils import Config, format_table_ref_for_bq
import time


@click.command()
@click.option("--fin", help="Source file.")
@click.option("--fout", help="Destination file.")
@monitor
def main(fin, fout):
    config = Config()
    client = bq.Client(project=config.project_id)
    table_ref = client.dataset(config.dataset_id).table("tmp")
    load_job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    tmp = np.loadtxt(fin, dtype="str")
    df = pd.DataFrame(data=tmp, columns=["pub_num"])
    client.load_table_from_dataframe(
        df, destination=table_ref, job_config=load_job_config
    )

    query = """
    SELECT 
      p.publication_number,
      tmp.pub_num
    FROM
      `patents-public-data.patents.publications` as p,
      {} as tmp
    WHERE
      REGEXP_EXTRACT(LOWER(p.publication_number), r'[a-z]+-(\d+)-[a-z0-9]+')=tmp.pub_num 
      AND p.country_code="US"
      """.format(
        format_table_ref_for_bq(table_ref)
    )
    time.sleep(
        3
    )  # otherwise we send the request before the effective completion of the table
    df = client.query(query).to_dataframe()
    df["publication_number"].to_frame().set_index("publication_number").to_csv(
        fout
    )


if __name__ == "__main__":
    main()

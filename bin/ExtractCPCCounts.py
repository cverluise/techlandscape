import click
from google.cloud import bigquery as bq
from wasabi import Printer


def extract_cpc_counts(
    project_id, dataset_id, table_id, country_list, date_low
):
    client = bq.Client(project=project_id)
    table_ref = client.dataset(dataset_id=dataset_id).table(table_id)

    country_list_prep = list(map(lambda x: '"' + str(x) + '"', country_list))
    query = """
    SELECT
      cpcs.code,
      COUNT(cpcs.code) AS counts
    FROM
      `patents-public-data.patents.publications` AS p,
      UNNEST(cpc) AS cpcs
    WHERE
      cpcs.code != ''
      AND country_code in ({})
      AND publication_date>={}
    GROUP BY cpcs.code
    """.format(
        ",".join(country_list_prep), date_low
    )

    query_config = bq.QueryJobConfig(
        write_disposition="WRITE_TRUNCATE", destination=table_ref
    )
    client.query(query=query, job_config=query_config).result()

    query = """
        SELECT
          'nb_patents' as code,
          COUNT(DISTINCT(publication_number)) AS counts
        FROM
          `patents-public-data.patents.publications` AS p,
          UNNEST(cpc) AS cpcs
        WHERE
          cpcs.code != ''
          AND country_code in ({})
          AND publication_date>={}
        """.format(
        ",".join(country_list_prep), date_low
    )

    query_config = bq.QueryJobConfig(
        write_disposition="WRITE_APPEND", destination=table_ref
    )
    client.query(query=query, job_config=query_config).result()


def load_cpc_counts_to_gs(project_id, dataset_id, table_id, bucket_id):
    client = bq.Client(project=project_id)
    destination_uri = "gs://{}/{}".format(bucket_id, table_id + ".csv.gz")
    table_ref = client.dataset(dataset_id=dataset_id).table(table_id)
    extract_config = bq.ExtractJobConfig(
        destination_format="CSV", compression="GZIP"
    )
    client.extract_table(
        table_ref, destination_uri, job_config=extract_config
    ).result()


@click.command()
@click.option("--project_id", default="brv-patent", help="gcp project id")
@click.option("--dataset_id", default="tech_landscape", help="bq dataset id")
@click.option("--table_id", default="cpc_counts", help="bq destination table")
@click.option(
    "--bucket_id", default="tech_landscape", help="bq destination table"
)
@click.option(
    "--country_list", default=["US", "FR", "DE", "GB"], help="country list"
)
@click.option("--date_low", default=19700000, help="lower bound")
@click.option("--verbose", default=True)
def main(
    project_id,
    dataset_id,
    table_id,
    bucket_id,
    country_list,
    date_low,
    verbose,
):
    if verbose:
        msg = Printer()
        msg.info(title="Extract CPC counts")
        msg.loading(text="Extracting")
    extract_cpc_counts(
        project_id, dataset_id, table_id, country_list, date_low
    )
    if verbose:
        msg.info(title="Done")
        msg.info(title="Load CPC counts to GS")
        msg.loading(text="Loading")
    load_cpc_counts_to_gs(project_id, dataset_id, table_id, bucket_id)
    if verbose:
        msg.info(title="Done")
        msg.divider("Good to know")
        msg.text(
            text=f"The cpc counts is now stored in your bucket. To download it in "
            f"your current directory, execute: \n"
            f"$gsutil cp gs://{bucket_id}/{table_id}.csv.gz ./"
        )


if __name__ == "__main__":
    main()

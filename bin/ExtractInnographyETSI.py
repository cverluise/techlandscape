import click
from google.cloud import bigquery as bq

from techlandscape.decorators import monitor
from techlandscape.utils import Config

config = Config()
client = bq.Client(project=config.project_id)


@click.command()
@click.option(
    "--path",
    default="data/seed/etsi.csv",
    help="File destination path. E.g: data/seed/etsi.csv.",
)
@monitor
def main(path):
    query = """
    SELECT
      *
    FROM
      `innography-174118.technical_standards.etsi`
      """
    df = client.query(query).to_dataframe()
    df.to_csv(path)


if __name__ == "__main__":
    main()

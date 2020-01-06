from techlandscape.config import Config
from google.cloud import bigquery as bq
from techlandscape.decorators import monitor
from wasabi import Printer
import click

msg = Printer()


def query_pubfam(config):
    return f"""
    SELECT
      publication_number,
      family_id
    FROM
      `patents-public-data.patents.publications`
      """


def query_abstract(config):
    return f"""
    SELECT
      family_id,
      ANY_VALUE(abstract) AS abstract
    FROM
      `patents-public-data.google_patents_research.publications` AS pub,
      `{config.project_id}.{config.dataset_id}.pubfam` AS pubfam
    WHERE
      pubfam.publication_number=pub.publication_number
      AND pub.abstract IS NOT NULL
      AND pub.abstract !=""
    GROUP BY
      family_id
      """


def query_country(config):
    return f"""
    SELECT
      family_id,
      SPLIT(STRING_AGG(DISTINCT(country))) AS country
    FROM
      `patents-public-data.google_patents_research.publications` AS pub,
      `{config.project_id}.{config.dataset_id}.pubfam` AS pubfam
    WHERE
      pubfam.publication_number=pub.publication_number
    GROUP BY
      family_id
      """


def query_meta_patents(config):
    return f"""
    SELECT
      family_id,
      MIN(publication_date) AS publication_date,
      SPLIT(STRING_AGG(DISTINCT(country_code))) AS country_code,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(cpc.code))) AS code)] AS cpc,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(ipc.code))) AS code)] AS ipc
    FROM
      `patents-public-data.patents.publications`,
      UNNEST(cpc) AS cpc,
      UNNEST(ipc) AS ipc
    GROUP BY
      family_id
    """


def query_citation(config):
    return f"""
    WITH
      tmp AS (
      SELECT
        pub.family_id,
        pub.publication_number,
        citation.publication_number AS citation
      FROM
        `patents-public-data.patents.publications` AS pub,
        UNNEST(pub.citation) AS citation)
    SELECT
      tmp.family_id,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(pubfam.family_id))) AS family_id,
        SPLIT(STRING_AGG(DISTINCT(pubfam.publication_number))) AS publication_number)] AS citation
    FROM
      tmp
    JOIN (
      SELECT
        *
      FROM
        `{config.project_id}.{config.dataset_id}.pubfam`) AS pubfam
    ON
      tmp.citation=pubfam.publication_number
    GROUP BY
      tmp.family_id
    """


def query_cited_by(config):
    return f"""
    WITH
      tmp AS (
      SELECT
        family_id,
        cited_by
      FROM (
        SELECT
          family_id,
          pub.publication_number,
          cited_by.publication_number AS cited_by
        FROM
          `patents-public-data.google_patents_research.publications` AS pub,
          UNNEST(pub.cited_by) AS cited_by
        JOIN (
          SELECT
            publication_number,
            family_id
          FROM
            `{config.project_id}.{config.dataset_id}.pubfam`) AS pubfam
        ON
          pubfam.publication_number=pub.publication_number) )
    SELECT
      tmp.family_id,
      [STRUCT(SPLIT(STRING_AGG(DISTINCT(pubfam.family_id))) AS family_id,
        SPLIT(STRING_AGG(DISTINCT(pubfam.publication_number))) AS publication_number)] AS cited_by
    FROM
      tmp
    JOIN (
      SELECT
        *
      FROM
        `{config.project_id}.{config.dataset_id}.pubfam`) AS pubfam
    ON
      tmp.cited_by=pubfam.publication_number
    GROUP BY
      tmp.family_id
    """


def query_patents_publications(config):
    return f"""
    SELECT
      family_id,
      citation,
      cpc,
      ipc,
      country_code
    FROM
      `{config.project_id}.{config.dataset_id}.citation` AS cit
    FULL OUTER JOIN (
      SELECT
        family_id AS fam_id_mp,
        cpc,
        ipc,
        country_code
      FROM
        `{config.project_id}.{config.dataset_id}.meta_patents`) AS mp
    ON
      cit.family_id = mp.fam_id_mp
    WHERE
      family_id IS NOT NULL
      """


def query_google_patents_research_publications(config):
    return f"""
    SELECT
      family_id,
      abstract,
      cited_by,
      country
    FROM
      `{config.project_id}.{config.dataset_id}.abstract` AS abs
    FULL OUTER JOIN (
      SELECT
        family_id AS fam_id_cit,
        cited_by,
        country
      FROM
        `{config.project_id}.{config.dataset_id}.cited_by` AS cit
      FULL OUTER JOIN (
        SELECT
          family_id AS fam_id_ctr,
          country
        FROM
          `{config.project_id}.{config.dataset_id}.country` ) AS ctr
      ON
        cit.family_id=ctr.fam_id_ctr) AS tmp
    ON
      abs.family_id = tmp.fam_id_cit
    WHERE
      family_id IS NOT NULL
    """


intermediary_tables = [
    "country",
    "abstract",
    "meta_patents",
    "citation",
    "cited_by",
]

table_id2query = {
    "pubfam": query_pubfam,
    "abstract": query_abstract,
    "country": query_country,
    "meta_patents": query_meta_patents,
    "citation": query_citation,
    "cited_by": query_cited_by,
    "google_patents_research_publications": query_google_patents_research_publications,
    "patents_publications": query_patents_publications,
}


def _create_dataset(dataset_id, config):
    dataset = bq.Dataset(f"{config.project_id}.{dataset_id}")
    config.client().create_dataset(dataset, exists_ok=True)
    msg.info(f"{dataset_id} created if not already existing.")


def _create_table(
    table_id, query, config, write_disposition, destination_ref=None
):
    """
    :param table_id:
    :param query: func, take config as input
    :param config:
    :param write_disposition:
    :param destination_ref: table_ref
    :return:
    """
    assert write_disposition in [
        "WRITE_EMPTY",
        "WRITE_TRUNCATE",
        "WRITE_APPEND",
    ]
    if not destination_ref:
        config.table_ref(table_id)

    job_config = bq.QueryJobConfig()
    job_config.write_disposition = write_disposition
    job_config.destination = destination_ref

    job = config.client().query(query(config), job_config)
    msg.info(
        f"{table_id} creation started. write_disposition is {write_disposition}"
    )
    return job


@monitor
def create_pubfam(config):
    _create_dataset(config.dataset_id, config)
    job = _create_table(
        "pubfam", table_id2query["pubfam"], config, "WRITE_TRUNCATE"
    )
    job.result()


def delete_intermediary_tables(config):
    for table_id in intermediary_tables:
        table = bq.Table(f"{config.project_id}.{config.dataset_id}.{table_id}")
        config.client().delete_table(table, not_found_ok=True)
        msg.info(f"{table_id} has been deleted.")
    msg.warn(
        f"Dataset {config.dataset_id} not deleted.",
        text="You can do it by hand from the GCP console or using the python "
        "API using client.delete_dataset() if needed",
    )


@monitor
def create_intermediary_tables(config):
    jobs = []
    for table_id in intermediary_tables:
        job = _create_table(
            table_id, table_id2query[table_id], config, "WRITE_TRUNCATE"
        )
        jobs += [job]
    jobs_done = list(map(lambda x: x.result(), jobs))


@monitor
def create_publications_tables(config):
    jobs = []
    table_id = "publications"
    for dataset_id in ["google_patents_research", "patents"]:
        _create_dataset(dataset_id, config)
        job = _create_table(
            table_id,
            table_id2query["_".join([dataset_id, table_id])],
            config,
            "WRITE_TRUNCATE",
            Config(
                project_id=config.project_id, dataset_id=dataset_id
            ).table_ref(table_id),
        )
        jobs += [job]
    jobs_done = list(map(lambda x: x.result(), jobs))


@click.command(help="Creates the data setting for family level expansion")
@click.option(
    "--tmp_config",
    default=None,
    help="Dataset where intermediary tables are created. E.g. my_project.my_dataset",
)
@click.option(
    "--clean",
    default=True,
    help="Bool. True if you want to discard intermediary tables at the end of the process (recommended).",
)
def main(tmp_config, clean):
    if tmp_config:
        project_id, dataset_id = tmp_config.split(".")
        config = Config(project_id=project_id, dataset_id=dataset_id)
        msg.info(f"Intermediary tables will be stored in {tmp_config}")
    else:
        config = Config()
        msg.info(
            f"Intermediary tables will be stored in {config.project_id}.{config.dataset_id}"
        )
    create_pubfam(config)
    create_intermediary_tables(config)
    create_publications_tables(config)
    if clean:
        delete_intermediary_tables(config)
        msg.info(f"Intermediary tables removed")


if __name__ == "__main__":
    main()

import time
from wasabi import Printer
from google.cloud import bigquery as bq

from techlandscape.expansion import (
    load_seed_to_bq,
    get_important_cpc,
    cpc_expansion,
    citation_expansion,
    draw_af_antiseed,
    draw_aug_antiseed,
    get_expansion_result,
)

# TODO make it executable from shell ?

msg = Printer()

client = bq.Client(project="brv-patent")

table_ref = client.dataset("tech_landscape").table("tmp")
load_job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
query_job_config = bq.QueryJobConfig(
    write_disposition="WRITE_APPEND", destination=table_ref
)

load_seed_to_bq(
    "data/test/hairdryer_sm.txt", client, table_ref, load_job_config
)
important_cpc = get_important_cpc(2, client, table_ref)
cpc_expansion(important_cpc, client, query_job_config)
with msg.loading("Ready, steady..."):
    time.sleep(3)  # it seems that some relief is required
msg.good("Go!")
citation_expansion("L1", client, table_ref, query_job_config)
with msg.loading("Ready, steady..."):
    time.sleep(3)  # it seems that some relief is required
msg.good("Go!")
citation_expansion("L2", client, table_ref, query_job_config)
draw_af_antiseed(1000, client, table_ref, query_job_config)
draw_aug_antiseed(1000, important_cpc, client, table_ref, query_job_config)
expansion_df = get_expansion_result(client, table_ref)

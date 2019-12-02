import asyncio

import pandas as pd

from expand import full_expansion
from prune import full_pruning
from infer import full_inference
from techlandscape.config import Config
from techlandscape.utils import get_uid
from techlandscape.expansion.io import load_to_bq
from techlandscape.pruning.prepare_data import get_train_test
from techlandscape.pruning.utils import cnn_params_grid

config = Config()
client = config.client()
table_name = get_uid(6)
data_path = "data/tmp/"
# models_path = "models/persist.nosync/"
# plots_path = "plots/persist.nosync/"

#############
# Expansion #
#############

full_expansion(
    seed_file="data/seed/generalplasticsurgery_unique.csv",
    table_name=table_name,
    data_path=data_path,
    pc_flavor="cpc",
    pc_counterfactual_f="data/pc/cpc_freq_cntyr.csv.gz",
    countries=["US"],
)

# WARNING: rows are unique at the ["publication_number", "expansion_level", "abstract"] level
# This is not the case at the "publication_number" level for example, this is almost the case at
# the ["publication_number", "expansion_level"] level
# TODO? drop duplicates L1-L2, keep only L1

###########
# Pruning #
###########

classif_df = pd.read_csv(
    f"{data_path}{table_name}_classif.csv.gz", index_col=0, compression="gzip"
)
texts_train, texts_test, y_train, y_test = get_train_test(classif_df)
# TODO work on reproducibility.
#  1. At this stage, model results might slightly change at each new
#  iteration
#  2. Save data vectors, check that the random_state actually does the work

segment_df = asyncio.run(
    full_pruning(
        table_name="hair_dryer",
        model_type="cnn",
        params_grid=cnn_params_grid,
        texts_train=texts_train,
        texts_test=texts_test,
        y_train=y_train,
        y_test=y_test,
        data_path=data_path,
        model_path=models_path,
    )
)

# TODO report on nb of occurences by expansion_level
# TODO error analysis (error_analysis.error_report(model, texts_test, x_test, y_test, .5))
# TODO check distribution of prediction proba!

tmp = classif_df.query("expansion_level=='SEED'")[
    ["publication_number", "expansion_level"]
]
tmp = tmp.append(
    segment_df.reset_index()[["publication_number", "expansion_level"]]
)
tmp = tmp.drop_duplicates(["publication_number"]).set_index(
    "publication_number"
)
load_to_bq(
    tmp,
    client=client,
    table_ref=config.table_ref(f"{table_name}_segment", client),
    job_config=config.load_job_config(),
)
# TODO? extend the pruned patents to family (single or expanded), call it backprop

#############
# Inference #
#############

full_inference(f"{table_name}_segment", data_path, plots_path)

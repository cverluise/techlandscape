import pandas as pd

from expand import full_expansion
from prune import full_pruning
from techlandscape.pruning.prepare_data import get_train_test
from techlandscape.pruning.utils import cnn_params_grid
from techlandscape.expansion.io import load_to_bq
from techlandscape.config import Config

config = Config()
client = config.client()
table_name = "hair_dryer"
data_path = "data/persist.nosync/"
model_path = "model/persist.nosync/"

#############
# Expansion #
#############

full_expansion(
    seed_file="data/seed/hair_dryer.csv",
    table_name=table_name,
    data_path=data_path,
    pc_flavor="cpc",
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

segment_df = full_pruning(
    table_name="hair_dryer",
    model_type="cnn",
    params_grid=cnn_params_grid,
    texts_train=texts_train,
    texts_test=texts_test,
    y_train=y_train,
    y_test=y_test,
    data_path=data_path,
    model_path=model_path,
)


# TODO report on nb of occurences by expansion_level
# TODO error analysis (error_analysis.error_report(model, texts_test, x_test, y_test, .5))
# TODO check distribution of prediction proba!

tmp = (
    segment_df.reset_index()
    .drop_duplicates(["publication_number"])[
        ["publication_number", "expansion_level"]
    ]
    .set_index("publication_number")
)
load_to_bq(tmp, config.table_ref(f"{table_name}_segment", client))
# TODO? extend the pruned patents to family (single or expanded), call it backprop

#############
# Inference #
#############

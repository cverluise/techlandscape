import os
import asyncio
import pandas as pd

from techlandscape.pruning import vectorize_text
from techlandscape.pruning.tune_model import get_best_model, grid_search
from techlandscape.expansion.io import get_expansion_result
from techlandscape.config import Config
from techlandscape.decorators import monitor, load_or_persist


# TODO add


@monitor
async def get_pruning_model(
    table_name,
    model_type,
    params_grid,
    texts_train,
    texts_test,
    y_train,
    y_test,
    data_path,
    model_path,
):
    """
    Return the best model given the <model_type> and the <params_grid>
    :return: (keras.model, List[str]), (model, texts_train)
    """
    assert model_type in ["cnn", "mlp"]
    assert os.path.isfile(
        os.path.join(data_path, f"{table_name}_classif.csv.gz")
    )
    assert os.path.exists(model_path)
    assert os.path.exists(data_path)

    log_root = os.path.join(model_path, table_name)
    #  f"{model_path}{table_name}"
    fio = os.path.join(
        model_path, f"{table_name}-{model_type}-performance.csv"
    )
    #  f"{model_path}{table_name}-{model_type}-performance.csv"

    @load_or_persist(fio=fio)
    def main():
        performance_df = grid_search(
            params_grid,
            model_type,
            texts_train,
            texts_test,
            y_train,
            y_test,
            log_root,
        )
        return performance_df

    performance_df = main()
    model = get_best_model(performance_df, model_type, log_root)
    return model, texts_train


async def get_pruning_data(client, table_ref, table_name, data_path):
    """
    Load expansion data
    """
    assert os.path.exists(data_path)
    assert table_ref.table_id == table_name

    fio = os.path.join(data_path, f"{table_name}_expansion.csv.gz")

    @load_or_persist(fio=fio)
    def main():
        expansion_df = get_expansion_result("*expansion", client, table_ref)
        return expansion_df

    main()


@monitor
def get_segment(
    model, model_type, texts_train, expansion_df, data_path, table_name
):
    """"""
    fio = f"{data_path}{table_name}_segment.csv"
    if os.path.isfile(fio):
        segment_df = pd.read_csv(fio, index_col=0)
    else:
        texts_expansion = expansion_df["abstract"].to_list()
        _, x_expansion, _ = vectorize_text.get_vectors(
            texts_train, texts_expansion, model_type
        )
        expansion_df["pred_score"] = model.predict_proba(x_expansion)
        # TODO: thresholding (tune_model.get_threshold)
        #  voraussetzungen
        #  1. train val test rather than train test
        #  2. think twice about the selection criteria
        segment_df = expansion_df.query("pred_score<.5")
        del expansion_df
        segment_df.to_csv(fio)
    return segment_df


async def full_pruning(
    table_name,
    model_type,
    params_grid,
    texts_train,
    texts_test,
    y_train,
    y_test,
    data_path,
    model_path,
    bq_config=None,
):
    """"""
    if os.path.isfile(f"{data_path}{table_name}_segment.csv"):
        segment_df = pd.read_csv(
            f"{data_path}{table_name}_segment.csv", index_col=0
        )
    else:
        config = bq_config if bq_config else Config()
        client = config.client()
        table_ref = config.table_ref(table_name)

        model_task = asyncio.create_task(
            get_pruning_model(
                table_name,
                model_type,
                params_grid,
                texts_train,
                texts_test,
                y_train,
                y_test,
                data_path,
                model_path,
            )
        )
        data_task = asyncio.create_task(
            get_pruning_data(client, table_ref, table_name, data_path)
        )

        await model_task
        await data_task

        model, texts_train = model_task.result()
        expansion_df = pd.read_csv(
            f"{data_path}{table_name}_expansion.csv.gz", compression="gzip"
        )

        segment_df = get_segment(
            model, model_type, texts_train, expansion_df, data_path, table_name
        )
    return segment_df

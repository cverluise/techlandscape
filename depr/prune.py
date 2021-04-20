import os
import asyncio
import pandas as pd

from techlandscape.pruning import vectorize_text
from techlandscape.pruning.tune_model import get_best_model, grid_search
from techlandscape.expansion.io import get_expansion_result
from techlandscape.config import Config
from techlandscape.decorators import monitor, load_or_persist

# TODO refactor? Only full_pruning here
CHUNK_SIZE = 2e5


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
    assert os.path.isfile(os.path.join(data_path, f"{table_name}_classif.csv.gz"))
    assert os.path.exists(model_path)
    assert os.path.exists(data_path)

    log_root = os.path.join(model_path, table_name)
    fio = os.path.join(model_path, f"{table_name}-{model_type}-performance.csv")

    @load_or_persist(fio=fio)
    def main():
        performance_df = grid_search(
            params_grid, model_type, texts_train, texts_test, y_train, y_test, log_root
        )
        return performance_df

    performance_df = main()
    model = get_best_model(performance_df, model_type, log_root)
    return model, texts_train


async def get_pruning_data(client, table_ref, table_name, data_path, countries=None):
    """
    Load expansion data
    """
    assert os.path.exists(data_path)
    assert table_ref.table_id == table_name

    fio = os.path.join(data_path, f"{table_name}_expansion.csv.gz")

    @load_or_persist(fio=fio)
    def main():
        expansion_df = get_expansion_result("*expansion", client, table_ref, countries)
        return expansion_df

    main()


@monitor
def get_segment(model, model_type, texts_train, expansion_df, y_train=None):
    """
    Return the set of patents classified in the 0 class
    :return: pd.DataFrame
    """
    # fio = os.path.join(data_path, f"{table_name}_segment.csv")

    texts_expansion = expansion_df["abstract"].to_list()
    _, x_expansion, _ = vectorize_text.get_vectors(
        texts_train, texts_expansion, model_type, y_train
    )
    expansion_df["pred_score"] = model.predict_proba(x_expansion)
    # TODO: thresholding (tune_model.get_threshold)
    #  voraussetzungen
    #  1. train val test rather than train test
    #  2. think twice about the selection criteria
    segment_df = expansion_df.query("pred_score<.5")
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
    countries=None,
    bq_config=None,
):
    """
    Implement the full pruning process. Return the segment (patents classified in the seed group)
    :return: pd.DataFrame
    """
    fio = os.path.join(data_path, f"{table_name}_segment.csv")

    if os.path.isfile(fio):  # @load_or_persist cannot be applied to a coroutine
        segment_df = pd.read_csv(fio, index_col=0)
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
            get_pruning_data(client, table_ref, table_name, data_path, countries)
        )

        await model_task
        await data_task

        model, _ = model_task.result()  # texts_train

        chunks = pd.read_csv(
            f"{data_path}{table_name}_expansion.csv.gz",
            compression="gzip",
            index_col=0,
            chunksize=CHUNK_SIZE,
        )
        segment_df = pd.DataFrame(
            columns=["publication_number", "expansion_level", "abstract"]
        )
        for chunk in chunks:
            segment_df = segment_df.append(
                get_segment(model, model_type, texts_train, chunk, y_train),
                ignore_index=True,
                sort=False,
            )
        segment_df.to_csv(fio)
    return segment_df

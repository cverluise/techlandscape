import itertools

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from wasabi import Printer

from techlandscape.pruning import train_model, vectorize_text


def get_threshold(model, x_test, y_test, step=0.02):
    """
    Return the decision threshold which maximizes the product of the f1_scores
    :param model: keras.engine.sequential.Sequential
    :param x_test: list
    :param y_test: list
    :param step: float
    :return: (float, float)
    """
    proba_test = model.predict(x_test)
    f1_prod = []
    range_ = np.arange(0, 1, step)
    for thresh in range_:
        pred_test = list(map(lambda x: 1 if x > thresh else 0, proba_test))
        _, _, fscore, support = precision_recall_fscore_support(y_test, pred_test)
        f1_prod += [fscore[0] * fscore[1]]
    threshs_star = range_[np.where(np.array(f1_prod) == max(f1_prod))]
    thresh_star = threshs_star[0] + (threshs_star[-1] - threshs_star[0]) / 2
    return thresh_star, f1_prod


def grid_search(
    params_grid, model_type, texts_train, texts_test, y_train, y_test, log_root
):
    """
    Return metrics of performance for all combinations of hyper-parameters in <params_grid>
    Log models in <log_root> as <log_root>-<model_type>-<model_name>
    :param params_grid: dict
    :param model_type: str, in ["cnn", "mlp"]
    :param texts_train: List[str]
    :param texts_test: List[str]
    :param y_train: List[int]
    :param y_test: List[int]
    :param log_root: str, e.g 'model/persist.nosync/hair_dryer'
    :return: pd.Dataframe, dataframe with model performances
    """
    assert isinstance(params_grid, dict)
    assert model_type in ["cnn", "mlp"]

    performance = {}
    _, x_test, _ = vectorize_text.get_vectors(
        texts_train, texts_test, model_type, y_train
    )

    for params_ in list(itertools.product(*list(params_grid.values()))):
        params = dict(zip(params_grid.keys(), params_))
        model_name = "-".join(
            list(
                map(
                    lambda x: x[0] + "_" + str(x[1]),
                    list(zip(params_grid.keys(), params_)),
                )
            )
        )
        if model_type == "cnn":
            # TODO assert keys params_grid
            model, history = train_model.train_cnn(
                texts_train, texts_test, y_train, y_test, params
            )
        else:
            # TODO assert keys params_grid
            model, history = train_model.train_mlp(
                texts_train, texts_test, y_train, y_test, params
            )
        pred_test = model.predict_classes(x_test)
        prec, rec, f1, support = precision_recall_fscore_support(y_test, pred_test)
        performance.update(
            {
                model_name: {
                    "prec_0": prec[0],
                    "prec_1": prec[1],
                    "rec_0": rec[0],
                    "rec_1": rec[1],
                    "f1_0": f1[0],
                    "f1_1": f1[1],
                    "support_0": support[0],
                    "support_1": support[1],
                }
            }
        )
        model.save(f"{log_root}-{model_type}-{model_name}.h5")

    performance_df = pd.DataFrame().from_dict(performance).T
    return performance_df


def get_best_model(performance_df, model_type, log_root):
    """
    Return the best model based on the product of the f1 score on the 0 and 1 class
    :param performance_df: pd.Dataframe, from grid_search
    :param model_type: str, in ["cnn", "mlp"]
    :param log_root: str, e.g 'model/persist.nosync/hair_dryer'
    :return: keras.model.Sequential
    """
    assert model_type in ["cnn", "mlp"]
    msg = Printer()
    performance_df["f1_prod"] = performance_df["f1_0"] * performance_df["f1_1"]
    best_model = list(performance_df.sort_values("f1_prod", ascending=False).index)[0]
    msg.info(f"Best model: {' '.join(best_model.split('-'))}")
    return load_model(f"{log_root}-{model_type}-{best_model}.h5")

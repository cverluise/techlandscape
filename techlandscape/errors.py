import numpy as np
from keras import Model
from typing import Tuple, List
import typer

app = typer.Typer()


@app.command()
def get_report(
    model: Model,
    texts_text: np.array,
    x_test: np.array,
    y_test: np.array,
    thresh_star: float,
) -> Tuple[List, List]:
    """
    Return the lists of false positives and false negatives (text)
    """

    def get_errors(
        model: Model, x_test: np.array, y_test: np.array, thresh_star: float
    ) -> Tuple[List, List, List]:
        """
        Return the index of false positives, false negatives and list of scores (all)
        """
        score = model.predict(x_test).flatten().tolist()
        pred_test = list(map(lambda x: 1 if x > thresh_star else 0, score))
        confusion = list(zip(y_test, pred_test))
        false_pos = [i for i, pair in enumerate(confusion) if pair == (0, 1)]
        false_neg = [i for i, pair in enumerate(confusion) if pair == (1, 0)]
        return false_pos, false_neg, score

    false_pos, false_neg, score = get_errors(
        model, x_test, y_test, thresh_star
    )
    false_pos_scores = np.array(score)[false_pos].tolist()
    false_pos_texts = np.array(texts_text)[false_pos].tolist()
    false_pos_report = list(zip(false_pos_scores, false_pos_texts))
    false_neg_scores = np.array(score)[false_neg].tolist()
    false_neg_texts = np.array(texts_text)[false_neg].tolist()
    false_neg_report = list(zip(false_neg_scores, false_neg_texts))
    return false_pos_report, false_neg_report


if __name__ == "__main__":
    app()

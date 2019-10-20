import numpy as np


def get_errors(model, x_test, y_test, thresh_star):
    """

    :param model:
    :param x_test:
    :param y_test:
    :param thresh_star:
    :return: (list, list, list), (false_pos, false_neg, pred_test)
    """
    score = model.predict(x_test).flatten().tolist()
    pred_test = list(map(lambda x: 1 if x > thresh_star else 0, score))
    confusion = list(zip(y_test, pred_test))
    false_pos = [i for i, pair in enumerate(confusion) if pair == (0, 1)]
    false_neg = [i for i, pair in enumerate(confusion) if pair == (1, 0)]
    return false_pos, false_neg, score


def error_report(model, texts_text, x_test, y_test, thresh_star):
    """

    :param model:
    :param texts_text:
    :param x_test:
    :param y_test:
    :param thresh_star:
    :return: List
    """
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

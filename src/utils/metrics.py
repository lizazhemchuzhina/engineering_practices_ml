from typing import Tuple

import numpy as np


def get_precision_recall_accuracy(
    y_pred: np.array, y_true: np.array
) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """

    recall = []
    precision = []
    classes = np.unique(np.concatenate([y_true, y_pred]))
    accuracy = 0
    count = 0
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1
    accuracy = count / len(y_true)
    for cur in classes:
        tp = 0
        fp = 0
        fn = 0
        for iteration_ in range(0, len(y_pred)):
            if y_true[iteration_] == cur and y_pred[iteration_] == cur:
                tp += 1
            if y_true[iteration_] != cur and y_pred[iteration_] == cur:
                fp += 1
            if y_true[iteration_] == cur and y_pred[iteration_] != cur:
                fn += 1
        precision.append(tp / (tp + fp) if tp + fp != 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn != 0 else 0)

    return np.array(precision), np.array(recall), accuracy

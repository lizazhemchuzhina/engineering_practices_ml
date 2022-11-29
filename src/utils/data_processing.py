from typing import Tuple
import math

import numpy as np
import pandas as pd


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters

    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    data = pd.read_csv(path_to_csv)
    shuffled_data = data.sample(frac=1)
    y = shuffled_data["label"].apply(lambda x: 1 if x == 'M' else 0)
    X = shuffled_data.drop(columns=["label"])
    return np.array(X), np.array(y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    data = pd.read_csv(path_to_csv)
    shuffled_data = data.sample(frac=1)
    y = shuffled_data["label"].apply(lambda x: x)
    X = shuffled_data.drop(columns=["label"])
    return np.array(X), np.array(y)


def train_test_split(X: np.array, y: np.array, ratio: float
                     ) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    size_train = math.floor(len(X) * ratio)
    return X[:size_train], y[:size_train], X[size_train:], y[size_train:]

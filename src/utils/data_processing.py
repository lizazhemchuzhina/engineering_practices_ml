import math
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import os


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters






    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    x_data : np.array
        Матрица признаков опухолей.
    y_data : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    data = pd.read_csv(path_to_csv)
    shuffled_data = data.sample(frac=1)
    y_data = shuffled_data["label"].apply(lambda x: 1 if x == "M" else 0)
    x_data = shuffled_data.drop(columns=["label"])
    return np.array(x_data), np.array(y_data)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    x_data : np.array
        Матрица признаков сообщений.
    y_data : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    data = pd.read_csv(path_to_csv)
    shuffled_data = data.sample(frac=1)
    y_data = shuffled_data["label"].apply(lambda x: x)
    x_data = shuffled_data.drop(columns=["label"])
    return np.array(x_data), np.array(y_data)


def train_test_split(
        x_data: np.array, y_data: np.array, ratio: float
) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    x_data : np.array
        Матрица признаков.
    y_data : np.array
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
    size_train = math.floor(len(x_data) * ratio)
    return (
        x_data[:size_train],
        y_data[:size_train],
        x_data[size_train:],
        y_data[size_train:],
    )


def main(input_, output_):
    X, y = read_cancer_dataset(input_)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    os.makedirs(output_, exist_ok=True)
    np.save(f"{output_}/X_train", X_train)
    np.save(f"{output_}/X_test", X_test)
    np.save(f"{output_}/y_train", y_train)
    np.save(f"{output_}/y_test", y_test)


if __name__ == "__main__":
    input_ = sys.argv[1]
    output_ = sys.argv[2]
    main(input_, output_)

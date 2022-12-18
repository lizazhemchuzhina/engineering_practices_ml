import os
import sys

import numpy as np

from src.utils.plots import plot_precision_recall, plot_roc_curve


def main(input_data="/home/lizazhemchuzhina/PycharmProjects/engineering_practices_ml/data/prepared"):
    X_train = np.load(f"{input_data}/X_train.npy")
    X_train = np.array(X_train)
    X_test = np.load(f"{input_data}/X_test.npy")
    X_test = np.array(X_test)
    y_train = np.load(f"{input_data}/y_train.npy")
    y_train = np.array(y_train)
    y_test = np.load(f"{input_data}/y_test.npy")
    y_test = np.array(y_test)
    mean = np.mean(np.array(X_train), axis=0)
    std = np.std(np.array(X_train), axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    plot_precision_recall(X_train, y_train, X_test, y_test)
    plot_roc_curve(X_train, y_train, X_test, y_test, 10)


if __name__ == '__main__':
    if not os.path.exists("../data"):
        os.mkdir("../data")
    input_ = sys.argv[1]
    main(input_)

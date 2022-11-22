import numpy as np
from src.utils.data_processing import read_cancer_dataset, train_test_split, read_spam_dataset
from src.utils.plots import plot_precision_recall, plot_roc_curve


def main():
    X, y = read_cancer_dataset("datasets/cancer.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    mean = np.mean(np.array(X_train), axis=0)
    std = np.std(np.array(X_train), axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    plot_precision_recall(X_train, y_train, X_test, y_test)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)

    X, y = read_spam_dataset("datasets/spam.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    mean = np.mean(np.array(X_train), axis=0)
    std = np.std(np.array(X_train), axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np

from model.KNearest_classifier import KNearest
from src.utils.metrics import get_precision_recall_accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, path, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k_neigh in ks:
        classifier = KNearest(k_neigh)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for class_ in range(classes):
            precisions[class_].append(precision[class_])
            recalls[class_].append(recall[class_])
        accuracies.append(acc)

    def plot(x_data, ys, ylabel, path_, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x_data[0], x_data[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x_data, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path_)
        plt.show()

    path_recall = path + "_recall.png"
    plot(ks, recalls, "Recall", path_recall)
    path_precision = path + "_prec.png"
    plot(ks, precisions, "Precision", path_precision)
    path_accur = path + "_acc.png"
    plot(ks, [accuracies], "Accuracy", path_accur, legend=False)


def plot_roc_curve(X_train, y_train, X_test, y_test, path, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k_neigh in ks:
        colors.append([k_neigh / ks[-1], 0, 1 - k_neigh / ks[-1]])
        knearest = KNearest(k_neigh)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for weight in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > weight else 1) for p in p_pred]
            tpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0)
                / positive_samples
            )
            fpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0)
                / (len(y_test) - positive_samples)
            )
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize=(7, 7))
    for tpr, fpr, color in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=color)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

from typing import NoReturn, List

import numpy as np

from model.KD_tree import KDTree


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, self.leaf_size)
        self.labels = y
        self.classes_number = len(np.unique(self.labels))

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """
        result = []
        for neighbors in self.tree.query(X, self.n_neighbors):
            class_probability = [0.0] * self.classes_number
            for i in range(self.classes_number):
                class_probability[i] = np.mean(np.array([int(self.labels[neighbor] == i) for neighbor in neighbors]))
            result.append(np.array(class_probability))
        return result

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(np.array(self.predict_proba(X)), axis=1)

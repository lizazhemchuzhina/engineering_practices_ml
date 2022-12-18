from typing import List, NoReturn

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

    def fit(self, x_fit: np.array, y_fit: np.array) -> NoReturn:
        """

        Parameters
        ----------
        x_fit : np.array
            Набор точек, по которым строится классификатор.
        y_fit : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(x_fit, self.leaf_size)
        self.labels = y_fit
        self.classes_number = len(np.unique(self.labels))

    def predict_proba(self, x_fit: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        x_fit : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """
        result = []
        for neighbors in self.tree.query(x_fit, self.n_neighbors):
            class_probability = [0.0] * self.classes_number
            for i in range(self.classes_number):
                class_probability[i] = np.mean(
                    np.array(
                        [int(self.labels[neighbor] == i) for neighbor in neighbors]
                    )
                )
            result.append(np.array(class_probability))
        return result

    def predict(self, x_pred: np.array) -> np.array:
        """

        Parameters
        ----------
        x_pred : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(np.array(self.predict_proba(x_pred)), axis=1)

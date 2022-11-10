from typing import List

import numpy as np


class Point:
    def __init__(self, index: int, datum: np.array):
        self.index = index
        self.datum = datum


class Node:
    def __init__(self, left_son, right_son, data, is_leaf):
        self.left_son = left_son
        self.right_son = right_son
        self.data = data
        self.is_leaf = is_leaf


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.leaf_size = leaf_size
        self.dimension = len(X[0])
        X = np.array([Point(i, datum) for i, datum in enumerate(X)])
        self.root = self.build(X, 0)

    def build(self, X: np.array, axis: int):
        sample_size = len(X)
        if sample_size <= 2 * self.leaf_size:
            return Node(None, None, X, True)
        X = np.array(sorted(X, key=lambda p: p.datum[axis]))
        return Node(
            left_son=self.build(X[:sample_size // 2], (axis + 1) % self.dimension),
            right_son=self.build(X[sample_size // 2 + 1:], (axis + 1) % self.dimension),
            data=X[sample_size // 2],
            is_leaf=False
        )

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        result = []
        for datum in X:
            result.append([point.index for point in self.find_knn(self.root, Point(-1, datum), k, 0)])
        return result

    def find_knn(self, node: Node, point: Point, k: int, axis: int) -> List[Point]:
        if node.is_leaf:
            return self.optimize(point, node.data, k)
        if node.data.datum[axis] < point.datum[axis]:
            first, second = node.right_son, node.left_son
        else:
            first, second = node.left_son, node.right_son
        temp = self.find_knn(first, point, k, (axis + 1) % self.dimension)
        max_distance = self.distance(point, temp[-1])
        if np.abs(point.datum[axis] - node.data.datum[axis]) < max_distance or len(temp) < k:
            temp += self.find_knn(second, point, k, (axis + 1) % self.dimension)
            temp += [node.data]
        return self.optimize(point, temp, k)

    def distance(self, first: Point, second: Point):
        return np.linalg.norm(first.datum - second.datum)

    def optimize(self, point: Point, others: List[Point], k: int) -> List[Point]:
        return sorted(others, key=lambda p: self.distance(p, point))[:k]

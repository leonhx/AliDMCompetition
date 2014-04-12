#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(data_path)
import prep

class LPM:
    """
    Latent Pattern Model

    Parameters
    ----------

    topN: int, optional(default=3526)
        results rank topN will be returned ONLY.

    rho: float, optional(default=0.5), [0, 1)
        penalty factor when bought two products in one day.
    """
    def __init__(self, topN=3526, rho=0.5, max_iterate=500):
        self.__N__ = topN
        assert 0 <= rho < 1
        self.__rho__ = rho
        self.__cached__ = {}
        self.__max_iter__ = max_iterate
    def fit(self, X):
        X = X[X[:, 2] == 1]
        self.__X__ = X
        self.__w__ = {} # weights of nodes
        for bi in np.unique(X[:, 1]):
            self.__w__[bi] = (X[:, 1] == bi).sum()
        self.__G__ = {} # Graph
        for ui in np.unique(X[:, 0]):
            u_data = X[X[:, 0] == ui]
            u_data = u_data[u_data[:, 3].argsort(axis=0)]
            u_data = u_data[:, [1, 3]]
            for i in range(len(u_data)):
                b1, t1 = u_data[i]
                self.__G__.setdefault(b1, {})
                for j in range(i+1, len(u_data)):
                    b2, t2 = u_data[j]
                    assert t2 >= t1
                    if t2 != t1:
                        self.__G__[b1].setdefault(b2, [])
                        self.__G__[b1][b2].append(t2 - t1)
                    elif b1 != b2:
                        self.__G__.setdefault(b2, {})
                        self.__G__[b1].setdefault(b2, [])
                        self.__G__[b2].setdefault(b1, [])
                        self.__G__[b1][b2].append(self.__rho__)
                        self.__G__[b2][b1].append(self.__rho__)
    def __paths__(self, src, dst, min_distance, max_distance):
        """>=min_distance, <=max_distance"""
        ps = set()
        for next_node in self.__G__[src]:
            for e in self.__G__[src][next_node]:
                if e == 0:
                    if dst == next_node and dst != src and e >= min_distance:
                        print '#', e
                        ps.add((src, e, next_node))
                    continue
                if e <= max_distance:
                    if dst == next_node and e >= min_distance:
                        print '#', e
                        ps.add((src, e, next_node))
                    next_ps = self.__paths__(next_node, dst, max(0, min_distance-e), max_distance-e)
                    for p in next_ps:
                        print '#', e
                        ps.add(tuple([src, e] + list(p)))
                    continue
        return ps
    def __path_score__(self, path):
        """
        path: (src, e1_w, n1, e2_w, n2, e3_w, ..., en_w, dst)
        """
        score = 0.
        for n in path[:-1:2]:
            score += 1. / self.__w__[n]
        return score
    def __propagate__(self, user_id, brand_id, min_distance, max_distance):
        if (brand_id, min_distance, max_distance) not in self.__cached__:
            pred = []
            Q = [(brand_id, min_distance, max_distance)]
            score = {brand_id: 1.0}
            valid = set()
            i = 0
            while Q and i < self.__max_iter__:
                i += 1
                bi, mind, maxd = Q[0]
                Q = Q[1:]
                for next_bi in self.__G__[bi]:
                    for e in self.__G__[bi][next_bi]:
                        if e <= maxd:
                            score.setdefault(next_bi, 0.)
                            score[next_bi] += score[bi] * 1. / self.__w__[bi]
                            if mind <= e:
                                valid.add(next_bi)
                            Q.append((next_bi, max(mind-e, 0), maxd-e))
            self.__cached__[brand_id, min_distance, max_distance] = []
            for bi in valid:
                self.__cached__[brand_id, min_distance, max_distance].append(
                    [user_id, bi, score[bi]])
        return self.__cached__[brand_id, min_distance, max_distance]
    def predict(self, time_now):
        brands = np.unique(self.__X__[:, 1])
        predictions = []
        for ui in np.unique(self.__X__[:, 0]):
            u_data = self.__X__[self.__X__[:, 0] == ui]
            for _, b1, _, t1 in u_data:
                predictions += self.__propagate__(ui, b1, time_now - t1, time_now - t1 + 30)
        predictions = np.array(predictions)
        predictions = predictions[predictions[:, 2].argsort(axis=0)[-self.__N__:][::-1]]
        print predictions[-1]
        return predictions[:, :2], predictions[:, 2]

def get_model():
    return LPM(topN=3526, rho=0, max_iterate=5)

if __name__ == '__main__':
    lpm = get_model()
    lpm.fit(all_data)
    lpm.__propagate__(47000, 1950, 24, 54)

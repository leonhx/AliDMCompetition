#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import math
import os

class UserCF:
    def __init__(self, similar='jaccard', topK=80):
        if similar == 'jaccard':
            self.__similarity__ = self.__jaccard__
        elif similar == 'cosine':
            self.__similarity__ = self.__cosine__
        else:
            raise ValueError('No such similarity method: %s' % similar)
        self.__K__ = topK
    def __jaccard__(self, train):
        """
        train:
        { user_id: {
                    item_id: score,
                    ...
                    }
        ...
        }
        """
        item_users = {}
        for u, items in train.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)

        C = {} # intersection
        for i, users in item_users.items():
            for u in users:
                C.setdefault(u, {})
                for v in users:
                    if u != v:
                        C[u].setdefault(v, 0)
                        C[u][v] += 1

        U = {} # union
        for u in train:
            U.setdefault(u, {})
            for v in train:
                if u != v:
                    U[u][v] = len(set(train[u].keys()).union(train[v].keys()))

        W = {}
        for u, related_users in C.items():
            W.setdefault(u, {})
            for v, cuv in related_users.items():
                W[u][v] = cuv*1. / U[u][v]

        return W
    def __cosine__(self, train):
        """
        train:
        { user_id: {
                    item_id: score,
                    ...
                    }
        ...
        }
        """
        item_users = {}
        for u, items in train.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)

        C = {}
        N = {}
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    if u != v:
                        C[u].setdefault(v, 0)
                        C[u][v] += 1

        W = {}
        for u, related_users in C.items():
            W.setdefault(u, {})
            for v, cuv in related_users.items():
                W[u][v] = cuv*1. / math.sqrt(N[u] * N[v])

        return W
    def fit(self, X):
        """
        X:
        numpy.array([
            [user, item, rating],
            ...
        ])
        or
        numpy.array([
            [user, item],
            ...
        ])
        """
        if X.shape[1] == 2:
            X = np.c_[X, np.ones((len(X),), dtype=X.dtype)]
        rank = {}
        train = {}
        for u, i, r in X:
            train.setdefault(u, {})
            train[u][i] = r
        W = self.__similarity__(train)
        for u in train:
            rank.setdefault(u, {})
            interacted_items = train[u].keys()
            for v, wuv in sorted(W[u].items(), key=lambda i: i[1], reverse=True)[:self.__K__]:
                for i, rvi in train[v].items():
                    if i not in interacted_items:
                        rank[u].setdefault(i, 0.0)
                        rank[u][i] += wuv * rvi

        self.__recomm__ = []
        self.__rating__ = []
        for u, items in rank.items():
            for i, r in items.items():
                self.__recomm__.append([u, i])
                self.__rating__.append(r)
        self.__recomm__ = np.array(self.__recomm__)
        self.__rating__ = np.array(self.__rating__)
    def predict(self, threshold=0):
        """
        parameter:
            threshold: only output the predictions of which the rating is
                       larger than threshold, 0 by default
        returns:
            numpy.array([
                [user, item],
                ...
            ]),
            numpy.array([
                rating,
                ...
            ])
        """
        choice = self.__rating__ > threshold
        return self.__recomm__[choice], self.__rating__[choice]

def extract_data(data):
    return data[data[:, 2] == 1, :2]

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data', 'train_data.npy')
    data = np.load(data_path)

    ucf = UserCF(similar='cosine', topK=60)
    # ucf = UserCF(topK=80)
    ucf.fit(extract_data(data))

    ub, r = ucf.predict(threshold=0.2)
    print(len(ub))

    pred_result = {}
    for ui, bi in ub:
        pred_result.setdefault(ui, set())
        pred_result[ui].add(bi)
    import pickle
    f = open(
        os.path.join(
            os.path.split(os.path.abspath(__file__))[0], 'pred_result.pkl'),
        'w')
    pickle.dump(pred_result, f)
    f.close()


#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import math
import os

class ItemCF:
    def __init__(self, penalty=None, topK=80, rankN=10):
        self.__K__ = topK
        self.__N__ = rankN
        if penalty:
            if penalty != 'iuf':
                raise ValueError('No such penalty method: %s' % penalty)
            self.__similarity__ = self.__cosine_iuf__
        else:
            self.__similarity__ = self.__cosine__
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
        C = {}
        N = {}
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items:
                    if i != j:
                        C[i].setdefault(j, 0)
                        C[i][j] += 1

        W = {}
        for i, related_items in C.items():
            W.setdefault(i, {})
            for j, cij in related_items.items():
                W[i][j] = cij*1. / math.sqrt(N[i] * N[j])

        return W
    def __cosine_iuf__(self, train):
        """
        train:
        { user_id: {
                    item_id: score,
                    ...
                    }
        ...
        }
        """
        C = {}
        N = {}
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items:
                    if i != j:
                        C[i].setdefault(j, 0)
                        C[i][j] += 1. / math.log(1 + len(items))

        W = {}
        for i, related_items in C.items():
            W.setdefault(i, {})
            for j, cij in related_items.items():
                W[i][j] = cij*1. / math.sqrt(N[i] * N[j])

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
            interacted_items = train[u]
            for i, pi in interacted_items.items():
                for j, wj in sorted(W[i].items(), key=lambda i: i[1], reverse=True)[:self.__K__]:
                    if j not in interacted_items:
                        rank[u].setdefault(j, 0.0)
                        rank[u][j] += wj * pi

        self.__recomm__ = []
        self.__rating__ = []
        for u, items in rank.items():
            for i, r in sorted(items.items(), key=lambda i: i[1], reverse=True)[:self.__N__]:
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

    icf = ItemCF(penalty='iuf')
    # ucf = UserCF(topK=80)
    icf.fit(extract_data(data))

    ub, r = icf.predict(threshold=0.1)
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


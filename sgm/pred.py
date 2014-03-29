#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import os

class STG:
    """STG is a directed bipartite graph G(U, S, I, E, w):
    U          - the set of user nodes,
    S          - the set of session nodes,
    I          - the set of item nodes,
    E          - the set of edges,
    w : E -> R - anon-negative weight function for edges
    N[u]       - all items viewed by the user u
    N[u, t]    - all items viewed by the user u at time t
    """
    def __init__(self, train, eta_u=1, eta_s=1):
        """train:
        numpy.array([
            [user_id, brand_id, type, visit_datetime],
            ...
        ])
eta_u:
        weight of edges from Item to User
eta_s:
        weight of edges from Item to Session
        """
        self.__G__ = {}
        self.__U__ = set()
        self.__S__ = set()
        self.__I__ = set()
        self.__E__ = set()
        self.__w__ = {}
        self.__N__ = {}
        for u, i, _, t in train:
            self.__U__.add(u)
            self.__S__.add((u, t))
            self.__I__.add(i)
            self.__E__.update([(u, i), (i, u), ((u, t), i), (i, (u, t))])
            self.__w__[u, i] = 1
            self.__w__[(u, t), i] = 1
            self.__w__[i, u] = eta_u
            self.__w__[i, (u, t)] = eta_s
            self.__N__.setdefault(u, set())
            self.__N__[u].add(i)
            self.__N__.setdefault((u, t), set())
            self.__N__[u, t].add(i)
            self.__G__.setdefault(u, {})
            self.__G__[u][i] = self.__w__[u, i]
            self.__G__.setdefault(i, {})
            self.__G__[i][u] = self.__w__[i, u]
            self.__G__[i][u, t] = self.__w__[i, (u, t)]
            self.__G__.setdefault((u, t), {})
            self.__G__[u, t][i] = self.__w__[(u, t), i]
    def G(self):
        return self.__G__
    def edge_weights(self):
        return self.__w__

class SGM:
    """Session-based Graph Model"""
    def __init__(self):
        pass
    def fit(self, X):
        """
        X:
        numpy.array([
            [user_id, brand_id, type, visit_datetime],
            ...
        ])
        """
        pass
    def predict(self):
        """
        parameter:
            threshold: only output the predictions of which the rating is
                       larger than threshold, 0 by default
        returns:
            numpy.array([
                [user, item],
                [user, item],
                ...
            ]),
            numpy.array([
                rating,
                rating,
                ...
            ])
        """
        pass

def extract_data(data):
    return data[data[:, 2] == 1]

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data', 'train_data.npy')
    data = np.load(data_path)

    sgm = SGM()
    sgm.fit(extract_data(data))

    ub, r = sgm.predict()
    ub = ub[r > 0.]
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


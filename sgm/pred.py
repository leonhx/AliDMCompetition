#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class STG:
    """STG is a directed bipartite graph G(U, S, I, E, w):
    U          - the set of user nodes,
    S          - the set of session nodes,
    I          - the set of item nodes,
    E          - the set of edges,
    w : E -> R - anon-negative weight function for edges
    N[u]       - all items viewed by the user u
    N[u, t]    - all items viewed by the user u at time t

    Parameters
    ----------
    train: numpy.array, column = [user_id, brand_id, type, visit_datetime]
        Training set.

    eta_u: int, float, optional (default=1)
        weight of edges from Item to User

    eta_s: int, float, optional (default=1)
        weight of edges from Item to Session

    time_interval: int, optional (default=3)
        each time_interval units will be considered as one time unit when
        construct session nodes
    """
    def __init__(self, train, eta_u=1, eta_s=1, time_interval=3):
        self.__G__ = {}
        self.__U__ = set()
        self.__S__ = set()
        self.__I__ = set()
        self.__E__ = set()
        self.__w__ = {}
        self.__N__ = {}
        for u, i, _, t in train:
            u = int(u)
            i = int(i)
            t = int(t)
            t = t/time_interval
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
    def out_degree(self, node):
        return len(self.__G__[node])
    def out(self, node):
        return set(self.__G__[node].keys())
    def users(self):
        return self.__U__
    def sessions(self):
        return self.__S__
    def items(self):
        return self.__I__

class SGM:
    """Session-based Graph Model

    Parameters
    ----------
    method: string, optional (default='ms-ipf')
        Method to make recommendations. Recently only 'ms-ipf' is available.

    topN: int, optional (default=20)
        unknown items ranking top-N will be recommended

    eta_u: int, float, optional (default=1)
        weight of edges from Item to User

    eta_s: int, float, optional (default=1)
        weight of edges from Item to Session
        eta = eta_u/eta_s is a parameter which controls the ratio of
        preferences (from an item node) to a user node against to a session
        node, thus affects the importance of long-term and short-term
        factors respectively on measuring item-item similarity.
        If eta -> inf, two items are only connected via user nodes and it
        means only users’ long-term preferences can contribute to item-item
        similarity.
        If eta = 0, two items are only connected via session nodes and thus
        only short-term preferences will contribute to item-item similarity.

    beta: int, float, between 0 and 1, optional (default=0.5)
        Weight of User nodes, while weight of User-Session nodes is 1-beta.
        beta is a parameter used to tune the ratio of injected preferences
        on the user node against the session node.
        beta = 0 means no preferences are injected into the user node; while
        beta = 1 means no preferences are injected into the session node.

    rho: int, float, between 0 and 1, optional (default=1)
        rho is a parameter to tune the impact of the out-degree in the
        propagation process.

    min_distance: int, optional (default=3), should be odd
        only paths whose distace is no smaller than min_distance will be
        considered

    time_interval: int, optional (default=3)
        each time_interval units will be considered as one time unit when
        construct session nodes
    """
    def __init__(self, method='ms-ipf', topN=20, eta_u=1, eta_s=1, beta=0.5, rho=1, min_distance=3, time_interval=3):
        if method == 'ms-ipf':
            self.__method__ = self.__ms_ipf__
        else:
            raise LookupError('No such method available: %s' % method)
        self.__topN__  = topN
        self.__eta_u__ = eta_u
        self.__eta_s__ = eta_s
        self.__beta__  = beta
        self.__rho__   = rho
        self.__min_distance__  = min_distance
        self.__time_interval__ = time_interval
    def __phi__(self, v, v_prime):
        eta = self.__eta_u__*1. / self.__eta_s__
        U = self.__stg__.users()
        S = self.__stg__.sessions()
        I = self.__stg__.items()
        out_v = self.__stg__.out(v)
        if (v in U or v in S) and v_prime in I:
            return 1. / len(out_v) ** self.__rho__
        if v in I and v_prime in U:
            return (eta / (eta*len(out_v.intersection(U)) + len(out_v.intersection(S)))) ** self.__rho__
        if v in I and v_prime in S:
            return 1. / (eta*len(out_v.intersection(U)) + len(out_v.intersection(S))) ** self.__rho__
    def __ms_ipf__(self, now_unit):
        """Multi-Source Injected Preference Fusion

        Parameters
        ----------
        now_unit: int
            time unit now, nomarlized to time interval already
        """
        self.__recomm__ = []
        self.__rating__ = []
        for u in self.__stg__.users():
            Q = [u]
            V = []
            distance = {u: 0}
            rank = {u: self.__beta__}
            if (u, now_unit) in self.__stg__.sessions():
                Q.append((u, now_unit))
                distance[u, now_unit] = 0
                rank[u, now_unit] = 1 - self.__beta__
            while Q:
                v = Q.pop()
                if distance[v] > self.__min_distance__:
                    break
                if v not in V:
                    V.append(v)
                    for v_prime in self.__stg__.out(v):
                        if v_prime not in V:
                            distance[v_prime] = distance[v] + 1
                            Q.append(v_prime)
                        if distance[v] < distance[v_prime]:
                            if v_prime not in rank:
                                if type(v_prime) == tuple:
                                    rank[v_prime] = 1 - self.__beta__
                                else:
                                    rank[v_prime] = self.__beta__
                            rank[v_prime] += rank[v] * self.__phi__(v, v_prime)
            rank = rank.items()
            rank.sort(key=lambda i: i[1], reverse=True)
            count = 0
            for i, r in rank:
                if count >= self.__topN__:
                    break
                if i in self.__stg__.items():
                    self.__recomm__.append([u, i])
                    self.__rating__.append(r)
                    count += 1
        self.__recomm__ = np.array(self.__recomm__)
        self.__rating__ = np.array(self.__rating__)
    def fit(self, X):
        """Fit the SGM model according to the given training data.

        Parameters
        ----------
        X: numpy.array, column = [user_id, brand_id, type, visit_datetime]
            Training set.
        """
        self.__stg__ = STG(extract_data(X), self.__eta_u__, self.__eta_s__,
            self.__time_interval__)
    def predict(self, now):
        """Perform recommendations

        Parameters
        ----------
        now: int
            time unit now, not nomarlized to time interval yet

        Returns
        -------
        pred: numpy.array, shape = [n_results, 2], column=[user_id, item_id]

        ratings: numpy.array, shape = [n_results]
            rating score of each corresponding prediction
        """
        try:
            return self.__recomm__, self.__rating__
        except AttributeError:
            self.__method__(now/self.__time_interval__)
            return self.__recomm__, self.__rating__

def extract_data(data):
    return data[data[:, 2] == 1]

def get_model():
    return SGM(method='ms-ipf', topN=20, eta_u=1, eta_s=1, beta=0.5, rho=1, min_distance=3, time_interval=3)

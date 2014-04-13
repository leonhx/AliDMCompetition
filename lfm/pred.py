#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(data_path)
import prep

class LFM:
    """
    Latent Factor Model

    Parameters
    ----------
    F: int
        number of latent factors

    alpha: float
        learning rate

    lamda: float
        regularization parameter

    ratio: int, float
        ratio of negative and positive samples

    scores: list/dict(any type support __getitem__) of int/float
        scores[i] is the score of type i(0 - click, 1 - buy, 2 - favo
        3 - cart)

    max_iterate: int
        max number of iterations
    """
    def __init__(self, F, alpha, lamda, ratio, scores, max_iterate):
        self.__F__ = F
        self.__alpha__ = alpha
        self.__lambda__ = lamda
        self.__ratio__ = ratio
        self.__scores__ = scores
        self.__max_iter__ = max_iterate
    def __extract__(self, train):
        new_X = []
        for ui in np.unique(train[:, 0]):
            u_data = train[train[:, 0] == ui]
            for bi in np.unique(u_data[:, 1]):
                ub_data = u_data[u_data[:, 1] == bi]
                score = 0.
                for _, _, action, time in ub_data:
                    score += self.__scores__[action]
                new_X.append([ui, bi, score])
        return new_X
    def __predict__(self, u, i):
        return sum(self.__P__[u][f] * self.__Q__[i][f] \
            for f in range(self.__F__))
    def __init_LFM__(self, train):
        import random, math
        self.__P__ = {}
        self.__Q__ = {}
        for u, i, rui in train:
            if u not in self.__P__:
                self.__P__[u] = [ random.random()/math.sqrt(self.__F__) \
                    for _ in range(self.__F__) ]
            if i not in self.__Q__:
                self.__Q__[i] = [ random.random()/math.sqrt(self.__F__) \
                    for _ in range(self.__F__) ]
    def fit(self, train):
        train = self.__extract__(train)
        self.__init_LFM__(train)
        alpha = self.__alpha__
        for _ in range(0, self.__max_iter__):
            print _
            for u, i, rui in train:
                pui = self.__predict__(u, i)
                eui = rui - pui
                for f in range(self.__F__):
                    self.__P__[u][f] += self.__alpha__ * (self.__Q__[i][f] * eui - self.__lambda__ * self.__P__[u][f])
                    self.__Q__[i][f] += self.__alpha__ * (self.__P__[u][f] * eui - self.__lambda__ * self.__Q__[i][f])
            alpha *= 0.9
    def predict(self, time_now):
        predictions = []
        for u in self.__P__:
            for i in self.__Q__:
                score = self.__predict__(u, i)
                predictions.append([u, i, score])
        predictions = np.array(predictions)
        predictions = predictions[predictions[:, 2].argsort(axis=0)[-3000:][::-1]]
        return predictions[:, :-1], predictions[:, -1]

def get_model():
    return LFM(F=20, alpha=0.02, lamda=0.01, ratio=10, scores=[1, 10, 5, 6], max_iterate=10)

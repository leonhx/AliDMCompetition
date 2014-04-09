#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(data_path)
import prep

class RawLR:
    """
    Parameters
    ----------
    model: An object which has interfaces `fit' and `predict'

    series: iterable of integers
        For example, [0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 21, 25, 29] means
        the time dimension should be splitted as 0, 1, 2, 3-4, 5-6, 7-8,
        9-11, 12-14, 15-17, 18-20, 21-24, 25-28, 29-
    """
    def __init__(self, model, series):
        self.__model__ = model
        assert len(series) > 0
        self.__series__ = series
    def fit(self, X):
        self.__X__ = X
    def predict(self, time_now):
        self.__now__ = time_now
        # ...
        return predictions, ratings
    def __extract__(self, X):
        for ui in np.unique(self.__X__[:, 0]):
            u_data = self.__X__[self.__X__[:, 0] == ui]
            for bi in np.unique(u_data[:, 1]):
                ub_data = u_data[u_data[:, 1] == bi]
                # ...

def get_model():
    pass

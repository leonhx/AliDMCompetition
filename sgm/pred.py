#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import os

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


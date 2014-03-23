#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath('./model1'))

import numpy as np

from reader import test_result, train_data
from predictor import pred_result

def f1(pred_result, test_result):
    """
    pred_result has the same format as test_result:
        key: user_id
        value: set of brand_id
    """
    pBrands = sum([len(pred_result[ui]) for ui in pred_result])
    hitBrands = 0
    for ui in np.unique(train_data.user_id):
        if ui in pred_result and ui in test_result:
            hitBrands += len(pred_result[ui].intersection(test_result[ui]))
    bBrands = sum([len(test_result[ui]) for ui in test_result])
    p = hitBrands * 1. / pBrands
    assert 0 < p <= 1
    r = hitBrands * 1. / bBrands
    assert 0 < r <= 1
    return (p, r, 2.*p*r/(p+r))

if __name__ == '__main__':
    p, r, f = f1(pred_result, test_result)
    print('Precision: %f' % p)
    print('Recall:    %f' % r)
    print('F1 score:  %f' % f)

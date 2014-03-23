#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Test script of models.

*Require*: predictions must be saved as a pickle file under the directory
of the model, and the file name should be pred_result.pkl.

pred_result.pkl should be the pickle file of a dict which has the format
described below:
    key: user_id
    value: set of brand_id
"""

import numpy as np

def f1(pred_result, test_result, all_userids):
    """
    pred_result has the same format as test_result:
        key: user_id
        value: set of brand_id
    """
    pBrands = sum([len(pred_result[ui]) for ui in pred_result])
    hitBrands = 0
    for ui in np.unique(all_userids):
        if ui in pred_result and ui in test_result:
            hitBrands += len(pred_result[ui].intersection(test_result[ui]))
    bBrands = sum([len(test_result[ui]) for ui in test_result])
    p = hitBrands * 1. / pBrands
    assert 0 < p <= 1
    r = hitBrands * 1. / bBrands
    assert 0 < r <= 1
    return (p, r, 2.*p*r/(p+r))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        raise LookupError('Please specify model to test.')
    for model in sys.argv[1:]:
        import os
        current_dir = os.path.join(*os.path.split(os.path.abspath(__file__))[:-1])
        model_path = os.path.join(current_dir, model, 'pred_result.pkl')
        result_path = os.path.join(current_dir, 'data', 'test_result.pkl')
        train_data_path = os.path.join(current_dir, 'data', 'train_data.npy')
        import pickle
        f = open(model_path, 'r')
        pred_result = pickle.load(f)
        f.close()
        f = open(result_path, 'r')
        test_result = pickle.load(f)
        f.close()
        all_userids = np.load(train_data_path)[:, 0]
        p, r, f = f1(pred_result, test_result, all_userids)
        print('=============')
        print('Model:     %s' % model)
        print('Precision: %f' % p)
        print('Recall:    %f' % r)
        print('F1 score:  %f' % f)

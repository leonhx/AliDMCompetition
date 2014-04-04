#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A script to test model automatically or generate result file of the model.

Usage
-----
    $ python run.py test model-sgm model-lr model-lfm
    $ python run.py gen model-sgm

Arguments
---------

1:  command, 'test' or 'gen'.
    'test' - test the model
    'gen'  - generate result file of the model

2~: model name, i.e., the folder where corresponding pred.py lies
    *NOTE*: command `gen` accepts one model only

Note
----

There should be a global variable named `model` in pred.py, which is
a properly initialized model object
"""

import numpy as np

import sys
import os
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
import prep

all_data = np.load(os.path.join(data_path, 'raw_data.npy'))

def ndarray2dict(ndarray):
    result = {}
    for u, i in ndarray:
        result.setdefault(u, set())
        result[u].add(i)
    return result

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

def test():
    if len(sys.argv) < 3:
        raise LookupError('Please specify model to test.')
    for model in sys.argv[2:]:
        sys_path = sys.path
        model_path = os.path.join(current_dir, model)
        sys.path.append(model_path)
        import pred
        # TODO
        # all_userids = np.load(train_data_path)[:, 0]
        # p, r, f = f1(pred_result, test_result, all_userids)
        # print_test_result(model, p, r, f)
        sys.path = sys_path

def gen():
    sys_path = sys.path
    if len(sys.argv) < 3:
        raise LookupError('Please specify model result to transfer.')
    elif len(sys.argv) > 3:
        raise UserWarning('Only one model result can be transfered.')
    model_path = os.path.join(current_dir, sys.argv[2])
    sys.path.append(model_path)
    import pred
    pred.model.fit(all_data)
    pred_result = ndarray2dict(pred.model.predict())
    lines = []
    for u, items in pred_result.items():
        line = '{0}\t{1}\n'.format(u, ','.join([str(i) for i in items]))
        lines.append(line)
    target_path = os.path.join(current_dir, 'result{0}.txt'.format(date.today().day))
    f = open(target_path, 'w')
    f.writelines(lines)
    f.close()
    sys.path = sys_path

if __name__ == '__main__':
    if sys.argv[1] not in ['test', 'gen']:
        raise LookupError('Unknown command: %d' % sys.argv[1])
    eval('%s()' % sys.argv[1])

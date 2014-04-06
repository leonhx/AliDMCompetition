#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A script to validate model automatically or generate result file of the model.

Usage
-----
    $ python run.py val model-sgm model-lr model-lfm
    $ python run.py gen model-sgm

Arguments
---------

1:  command, 'val' or 'gen'.
    'val' - validate the model
    'gen'  - generate result file of the model

2~: model name, i.e., the folder where corresponding pred.py lies
    *NOTE*: command `gen` accepts one model only

Note
----

There should be a function named `get_model` in pred.py, which returns
a properly initialized model object.

A model object should have at least two methods: `fit` and `predict`.
`fit`: ndarray(column=[user_id, brand_id, type, visit_datetime]) -> None
`predict`: int (time now) -> (ndarray(column=[user_id, item_id]), ndarray(column = [rating]))
"""

import numpy as np
import pylab as pl

import sys
import os
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
sys.path.append(data_path)
import prep

all_data = np.load(os.path.join(data_path, 'raw_data.npy'))

def ndarray2dict(ndarray):
    result = {}
    for u, i in ndarray:
        result.setdefault(u, set())
        result[u].add(i)
    return result

def dict_size(dictionary):
    size = 0
    for i in dictionary:
        size += len(dictionary[i])
    return size

def f1(pred_result, val_result, all_userids):
    """
    pred_result has the same format as val_result:
        key: user_id
        value: set of brand_id
    """
    pBrands = sum([len(pred_result[ui]) for ui in pred_result])
    hitBrands = 0
    for ui in all_userids:
        if ui in pred_result and ui in val_result:
            hitBrands += len(pred_result[ui].intersection(val_result[ui]))
    bBrands = sum([len(val_result[ui]) for ui in val_result])
    p = hitBrands * 1. / pBrands
    assert 0 < p <= 1
    r = hitBrands * 1. / bBrands
    assert 0 < r <= 1
    return p, r, 2.*p*r/(p+r)

def print_result(pred_result, val_result, description, p, r, f):
    pred_count = dict_size(pred_result)
    real_count = dict_size(val_result)
    assert round(pred_count * p) == round(real_count * r)
    print('****** %s' % description)
    print('Precision: {:f}%'.format(p*100))
    print('Recall:    {:f}%'.format(r*100))
    print('F1 Score:  {:f}%'.format(f*100))
    print('# Pred:    %d' % pred_count)
    print('# Real:    %d' % real_count)
    print('# Hit:     %d' % round(pred_count*p))

def plot_result(model_name, val_cases, Ps, Rs, Fs):
    pl.figure(model_name)
    pl.plot(range(len(val_cases)), Ps)
    pl.plot(range(len(val_cases)), Rs)
    pl.plot(range(len(val_cases)), Fs)
    pl.legend(('Precision', 'Recall', 'F1 Score'))
    pl.xticks(range(len(val_cases)), [i[3] for i in val_cases])
    pl.show()

def val():
    if len(sys.argv) < 3:
        raise LookupError('Please specify model to val.')
    val_cases = [
        (prep.date(5, 18), prep.date(6, 17), 'predicts 5/18 to 6/17', '5/18'),
        (prep.date(6, 17), prep.date(7, 17), 'predicts 6/17 to 7/17', '6/17'),
        (prep.date(7, 17), prep.date(8, 16), 'predicts 7/17 to 8/16', '7/17'),
    ]
    for model_name in sys.argv[2:]:
        sys_path = sys.path[:]
        model_path = os.path.join(current_dir, model_name)
        sys.path.append(model_path)
        import pred
        reload(pred)
        print('======')
        print('Model:\t%s' % model_name)
        Ps = []
        Rs = []
        Fs = []
        for TRAIN_DATE, TEST_DATE, DESC, _ in val_cases:
            model = pred.get_model()
            train_data = all_data[all_data[:, 3] < TRAIN_DATE]
            model.fit(train_data)
            predictions, _ = model.predict(TRAIN_DATE-1)
            pred_result = ndarray2dict(predictions)
            val_data = all_data[np.logical_and(
                all_data[:, 2] == 1,
                np.logical_and(
                    all_data[:, 3] >= TRAIN_DATE,
                    all_data[:, 3] < TEST_DATE
                )
            )]
            val_result = ndarray2dict(val_data[:, :2])
            all_userids = np.unique(train_data[:, 0])
            p, r, f = f1(pred_result, val_result, all_userids)
            Ps.append(p)
            Rs.append(r)
            Fs.append(f)
            print_result(pred_result, val_result, DESC, p, r, f)
        plot_result(model_name, val_cases, Ps, Rs, Fs)
        sys.path = sys_path

def gen():
    sys_path = sys.path[:]
    if len(sys.argv) < 3:
        raise LookupError('Please specify model result to transfer.')
    elif len(sys.argv) > 3:
        raise UserWarning('Only one model result can be transfered.')
    model_path = os.path.join(current_dir, sys.argv[2])
    sys.path.append(model_path)
    import pred
    model = pred.get_model()
    model.fit(all_data)
    predictions, _ = model.predict(prep.date(8, 15))
    pred_result = ndarray2dict(predictions)
    print('Get %d predictions' % dict_size(pred_result))
    lines = []
    for u, items in pred_result.items():
        line = '{0}\t{1}\n'.format(u, ','.join([str(i) for i in items]))
        lines.append(line)
    from datetime import date
    target_path = os.path.join(current_dir, 'result%02d.txt'%date.today().day)
    f = open(target_path, 'w')
    f.writelines(lines)
    f.close()
    sys.path = sys_path

if __name__ == '__main__':
    if sys.argv[1] not in ['val', 'gen']:
        raise LookupError('Unknown command: %d' % sys.argv[1])
    eval('%s()' % sys.argv[1])

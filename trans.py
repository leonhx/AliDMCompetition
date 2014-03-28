#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

from datetime import date

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise LookupError('Please specify model result to transfer.')
    elif len(sys.argv) > 2:
        raise UserWarning('Only one model result can be transfered.')
    current_dir = os.path.split(os.path.abspath(__file__))[0]
    model_path = os.path.join(current_dir, sys.argv[1], 'pred_result.pkl')
    target_path = os.path.join(current_dir, 'result{0}.txt'.format(date.today().day))
    import pickle
    f = open(model_path, 'r')
    pred_result = pickle.load(f)
    f.close()
    lines = []
    for u, items in pred_result.items():
        line = '{0}\t{1}\n'.format(u, ','.join([str(i) for i in items]))
        lines.append(line)
    f = open(target_path, 'w')
    f.writelines(lines)
    f.close()

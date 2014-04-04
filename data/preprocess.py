#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

data_dir = os.path.dirname(os.path.abspath(__file__))
csv_data_path = os.path.join(data_dir, 't_alibaba_data.csv')
raw_data_path = os.path.join(os.path.join(data_dir, 'raw_data.npy'))

def date(month, day):
    import datetime
    YEAR = 2013
    START_DATE = datetime.date(YEAR, 1, 1)
    return (datetime.date(YEAR, month, day) - START_DATE).days

def date_parser(s):
    import re
    DATE_RE = re.compile(u'(\d+)月(\d+)日')
    m = DATE_RE.match(s.decode('gbk'))
    return date(int(m.group(1)), int(m.group(2)))

if __name__ == '__main__':
    import numpy as np
    raw_data = np.loadtxt(csv_data_path, dtype=int, delimiter=',', converters={3: date_parser}, skiprows=1)
    np.save(raw_data_path, raw_data)

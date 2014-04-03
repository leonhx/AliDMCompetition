#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import re
import os

YEAR = 2013
START_DATE = datetime.date(YEAR, 1, 1)
DATE_RE = re.compile(u'(\d+)月(\d+)日')
current_dir = os.path.split(os.path.abspath(__file__))[0]
csv_data_path = os.path.join(current_dir, 't_alibaba_data.csv')
raw_data_path = os.path.join(os.path.join(current_dir, 'raw_data.npy'))

def date_parser(s):
    m = DATE_RE.match(s.decode('gbk'))
    date_delta = datetime.date(YEAR, int(m.group(1)), int(m.group(2))) - START_DATE
    return date_delta.days

if __name__ == '__main__':
    raw_data = np.loadtxt(csv_data_path, dtype=int, delimiter=',', converters={3: date_parser}, skiprows=1)
    np.save(raw_data_path, raw_data)

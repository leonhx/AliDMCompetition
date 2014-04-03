#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A script to test model automatically or generate result file of the model.

Usage
-----
    $ python run.py test model-sgm
    $ python run.py gen model-sgm

Arguments
---------

1: 'test' or 'gen'.
    'test' - test the model
    'gen'  - generate result file of the model

2: model name, i.e., the folder where corresponding pred.py lies

Note
----

There should be a global variable named `model` in pred.py, which is
a properly initialized model object
"""

import numpy as np

current_dir = os.path.split(os.path.abspath(__file__))[0]

import sys
import os
import datetime

sys.path.append(os.path.join(current_dir, 'data'))
import preprocess as pre

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

from pyrimidine.utils import *

def test_pattern():
    assert pattern([[0,1,1],[1,1,0]]) == '*1*'

#!/usr/bin/env python3


from pyrimidine.utils import *

def test_pattern():
    assert pattern([[0,1,1],[1,1,0]]) == '*1*'


def test_rotation():
    res = rotation([1,2,3,4,5], [2,4,3,5,1])
    assert res == [(0, 1, 3, 4)]

    res = rotation([5,2,3,1,4], [2,5,3,4,1])
    assert res == [(0, 1), (3, 4)]

    assert [2,5,3,4,1] == permutate([5,2,3,1,4], res)


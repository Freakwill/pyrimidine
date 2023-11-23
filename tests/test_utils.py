#!/usr/bin/env python3


from pyrimidine.utils import *


class TestUtils(unittest.TestCase):

    def test_pattern(self):
        assert pattern([[0,1,1],[1,1,0]]) == '*1*'

    def test_rotation(self):
        res = rotation([1,2,3,4,5], [2,4,3,5,1])
        assert res == [(0, 1, 3, 4)]

        res = rotation([5,2,3,1,4], [2,5,3,4,1])
        assert res == [(0, 1), (3, 4)]

        assert [2,5,3,4,1] == permutate([5,2,3,1,4], res)

    def test_rand(self):
        i, j = randint2()
        assert i < j


#!/usr/bin/env python3


from pyrimidine.utils import *


class TestUtils:

    def test_pattern(self):
        assert pattern([[0,1,1], [1,1,0]]) == '*1*'

    def test_rotation(self):
        res = rotations([1,2,3,4,5], [2,4,3,5,1])
        assert res == [(0, 1, 3, 4)]

        res = rotations([5,2,3,1,4], [2,5,3,4,1])
        assert res == [(0, 1), (3, 4)]

        assert [2,5,3,4,1] == rotate([5,2,3,1,4], res)

    def test_rand(self):
        i, j = randint2()
        assert i < j

    def test_prufer(self):
        x = [2,5,6,8,2,5]
        assert prufer_decode(x) == [(1, 2), (3, 5), (4, 6), (6, 8), (7, 2), (2, 5), (5, 8)]


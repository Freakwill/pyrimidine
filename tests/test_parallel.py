#!/usr/bin/env python3


from pyrimidine.parallel import *

import pytest


@pytest.fixture
def example():
    @dask_apply
    class C:

        def __init__(self, v=[1,2,3]):
            self.v = v
            self._map = MTMap(type_=tuple)

        @property
        def map(self):
            return self._map

        @map.setter
        def map(self, v):
            self._map = v

        def __iter__(self):
            return iter(self.v)

    return C

class TestParallel:

    def test_mt(self, example):
        c = example([1,2,3,4])
        assert c.map(lambda x:x+1, c) == (2, 3, 4, 5)
        assert c.apply(lambda x:x+2) == [3, 4, 5, 6]

    def test_dask(self, example):
        c = example([2,3,4,5])
        c.map = DaskMap(type_=tuple)
        assert c.map(lambda x:x-1, c) == (1, 2, 3, 4)


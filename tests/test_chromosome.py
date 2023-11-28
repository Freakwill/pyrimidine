#!/usr/bin/env python3


import numpy as np
from pyrimidine.chromosome import *


class TestChromosome:

    def test_random(self):
        b = (BinaryChromosome // 10).random()
        assert len(b) == 10
        assert all(bi in {0,1} for bi in b)

    def test_cross(self):
        C = BinaryChromosome
        b1 = C.random()
        b2 = C.random()
        b = b1.cross(b2)
        assert isinstance(b, C) and b[0] == b1[0] and b[-1] == b2[-1]

    def test_mutate(self):
        C = BinaryChromosome
        b = C.random()
        l = len(b)
        b.mutate()
        assert isinstance(b, BinaryChromosome) and len(b) == l

    def test_quantum(self):
        q = QuantumChromosome.random()
        q.measure()
        assert q.measure_result.dtype == np.int_


#!/usr/bin/env python3


import numpy as np
from pyrimidine.individual import *


class TestChromosome:

    def test_make(self):
        b = makeIndividual(element_class=BinaryChromosome, n_chromosomes=2, size=8).random()
        assert len(b) == 2

    def test_make_binary(self):
        b = makeBinaryIndividual(size=(8, 4)).random()
        assert len(b[0]) == 8 and len(b[1]) == 4

    def test_cross(self):
        I = MonoIndividual[BinaryChromosome]
        b1 = I.random()
        b2 = I.random()
        b = b1.cross(b2)
        assert isinstance(b, I)

    def test_mutate(self):
        I = MonoIndividual[BinaryChromosome]
        b = I.random()
        l = len(b[0])
        b.mutate()
        assert isinstance(b, I) and isinstance(b[0], BinaryChromosome) and len(b[0]) == l


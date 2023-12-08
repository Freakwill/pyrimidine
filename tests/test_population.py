#!/usr/bin/env python3


import numpy as np
from pyrimidine.population import *
from pyrimidine.individual import *


class TestChromosome:

    def test_random(self):
        p = (StandardPopulation[makeBinaryIndividual()] // 10).random()
        assert len(p) == 10
        assert all(bi in {0,1} for bi in p[0][0])

    def test_mate(self):
        p = (StandardPopulation[makeBinaryIndividual()] // 10).random()
        p.mate()
        assert True

    def test_clone(self):
        p = (StandardPopulation[makeBinaryIndividual()] // 10).random()
        pp = p.clone()
        assert isinstance(pp, StandardPopulation) and len(pp) == 10

    def test_create(self):
        MyIndividual = (BinaryChromosome // 10).set_fitness(lambda o:sum(o.decode()))
        MyPopulation = HOFPopulation[MyIndividual] // 16
        assert MyPopulation.element_class == MyIndividual


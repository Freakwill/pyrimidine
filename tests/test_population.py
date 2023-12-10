#!/usr/bin/env python3


from pyrimidine.population import *
from pyrimidine.individual import *

import pytest


@pytest.fixture
def example_population():
    return StandardPopulation[binaryIndividual()] // 10


class TestPopultion:

    def test_random(self, example_population):
        p = example_population.random()
        assert len(p) == 10
        assert all(bi in {0,1} for bi in p[0][0])

    def test_mate(self, example_population):
        p = example_population.random()
        p.mate()
        assert True

    def test_merge(self, example_population):
        p = example_population.random()
        q = example_population.random()
        p.merge(q)
        assert len(p) == 20

    def test_clone(self, example_population):
        p = example_population.random()
        pp = p.clone()
        assert isinstance(pp, StandardPopulation) and len(pp) == 10

    def test_create(self):
        MyIndividual = (BinaryChromosome // 10).set_fitness(sum)
        MyPopulation = HOFPopulation[MyIndividual] // 16
        assert MyPopulation.element_class == MyIndividual


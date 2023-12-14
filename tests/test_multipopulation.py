#!/usr/bin/env python3


from pyrimidine.multipopulation import MultiPopulation
from pyrimidine.population import StandardPopulation
from pyrimidine.individual import MonoIndividual
from pyrimidine.chromosome import BinaryChromosome

import pytest


@pytest.fixture
def example_population():
    class _Individual(MonoIndividual[BinaryChromosome // 10]):

        def decode(self):
            return self[0]

        def _fitness(self):
            return _evaluate(self.decode())

    class _Population(StandardPopulation):
        element_class = _Individual
        default_size = 10

    class _MultiPopulation(MultiPopulation):
        element_class = _Population
        default_size = 2

    return _MultiPopulation


class TestPopultion:

    def test_random(self, example_population):
        p = example_population.random()
        assert len(p) == 2 and len(p.individuals) == 20

    def test_mate(self, example_population):
        p = example_population.random()
        p.mate()
        assert True

    def test_migrate(self, example_population):
        p = example_population.random()
        p.migrate()
        assert len(p) == 2

    def test_clone(self, example_population):
        p = example_population.random()
        pp = p.clone()
        assert isinstance(pp[0], StandardPopulation) and isinstance(pp[1], StandardPopulation)


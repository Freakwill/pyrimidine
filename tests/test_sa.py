#!/usr/bin/env python3


from pyrimidine.individual import MonoIndividual
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.local_search.simulated_annealing import SimulatedAnnealing

from pyrimidine.benchmarks.special import rosenbrock

import pytest


@pytest.fixture(scope="class")
def example():
    n = 20
    f = rosenbrock

    class MyIndividual(SimulatedAnnealing, MonoIndividual):
        element_class = FloatChromosome.set(default_size=n)

        def _fitness(self):
            return - f(self.chromosome)

        def get_neighbour(self):
            cpy = self.clone()
            cpy.mutate()
            # or cpy.chromosomes[0] = cpy.chromosome.random_neighbour()
            return cpy

    return MyIndividual


class TestSA:

    def test_evolve(self, example):
        I = example
        self.individual = I.random()
        data = self.individual.evolve(max_iter=3, history=True)
        assert len(data) == 4


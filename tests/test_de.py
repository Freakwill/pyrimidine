#!/usr/bin/env python3


from pyrimidine.individual import MonoIndividual
from pyrimidine.population import HOFPopulation, BasePopulation
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.de import DifferentialEvolution

from pyrimidine.benchmarks.special import rosenbrock

import pytest

@pytest.fixture(scope="class")
def pop():
    n = 20
    f = rosenbrock

    class MyIndividual(MonoIndividual):
        element_class = FloatChromosome // n

        def _fitness(self):
            return -f(self.chromosome)

    class _Population1(DifferentialEvolution, BasePopulation):
        element_class = MyIndividual
        default_size = 10

    class _Population2(HOFPopulation):
        element_class = MyIndividual
        default_size = 10

    # _Population2 = HOFPopulation[MyIndividual] // 10

    return _Population1, _Population2


class TestDE:

    @classmethod
    def setup_class(cls):
        cls.Populations = cls.request.getfixturevalue("pop")
    
    def test_clone(self):
        P1, P2 = cls.Populations
        p2 = P1.random().clone(type_=P2)
        assert isinstance(p2, P2)

    def test_evolve(self):
        P1, P2 = cls.Populations
        self.population1 = P1.random()
        self.population2 = P2.random()
        stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
        data1 = self.population1.evolve(stat=stat, n_iter=5, history=True)
        data2 = self.population2.evolve(stat=stat, n_iter=5, history=True)
        assert ('Mean Fitness' in data1.columns and 'Best Fitness' in data1.columns and
            'Mean Fitness' in data2.columns and 'Best Fitness' in data2.columns)


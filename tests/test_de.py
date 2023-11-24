#!/usr/bin/env python3

import unittest

from pyrimidine.individual import MonoIndividual
from pyrimidine.population import HOFPopulation, BasePopulation
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.de import DifferentialEvolution

from pyrimidine.benchmarks.special import rosenbrock


class TestDE(unittest.TestCase):

    def setUp(self):

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
        self.Population1 = _Population1
        self.population1 = Population1.random()
        self.Population2 = _Population2
        self.population2 = Population2.random()

    def test_clone(self):
        self.population2 = self.population1.clone(type_=self.Population2) # population 2 with the same initial values to population 1
        assert isinstance(self.population2, self.Population2)

    def test_evolve(self):
        stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
        data1 = self.population1.evolve(stat=stat, n_iter=10, history=True)
        data2 = self.population2.evolve(stat=stat, n_iter=10, history=True)
        assert True


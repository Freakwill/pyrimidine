#!/usr/bin/env python3

import unittest

from pyrimidine import MonoIndividual, BinaryChromosome
from pyrimidine import StandardPopulation, HOFPopulation
from pyrimidine.benchmarks.optimization import *


class TestGA(unittest.TestCase):

    def setUp(self):

        n_bags = 10
        _evaluate = Knapsack.random(n_bags)

        class MyIndividual(MonoIndividual):

            element_class = BinaryChromosome.set(default_size=n_bags)

            def _fitness(self) -> float:
                return _evaluate(self.chromosome)

        class MyPopulation(StandardPopulation):
            element_class = MyIndividual
            default_size = 8

        self.MyPopulation = MyPopulation

        class YourPopulation(HOFPopulation):
            element_class = MyIndividual
            default_size = 8

        self.YourPopulation = YourPopulation
        self.population = self.MyPopulation.random()

    def test_random(self):
        self.population = self.MyPopulation.random()
        cpy = self.population.clone()
        self.population.merge(cpy, n_sel=8)
        assert len(self.population) == 16

    def test_evolve(self):
        self.population.evolve(n_iter=2)
        assert True
    
    def test_stat(self):
        stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
        data = self.population.evolve(stat=stat, n_iter=3, history=True)
        assert 'mean_fitness' in data.columns and 'best_fitness' in data.columns


    def test_hof(self):
        population = self.YourPopulation.random()

        stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
        data = population.evolve(stat=stat, n_iter=5, history=True)

        def increasing(x):
            all(xi <= xj for xi, xj in zip(x[:-1], x[1:]))

        assert increasing(data['best_fitness'])
    
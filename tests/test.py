#!/usr/bin/env python3

from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
from pyrimidine.benchmarks.optimization import *

n_bags = 10
_evaluate = Knapsack.random(n_bags)  # : 0-1 array -> float

def test_random():
    class MyIndividual(MonoIndividual):
        element_class = BinaryChromosome.set(default_size=n_bags)
        def _fitness(self) -> float:
            return _evaluate(self.chromosome)

    class MyPopulation(StandardPopulation):
        element_class = MyIndividual
        default_size = 8

    pop = MyPopulation.random()

    cpy = pop.clone()
    pop.merge(cpy, n_sel=8)

    assert len(pop) == 16


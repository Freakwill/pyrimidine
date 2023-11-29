#!/usr/bin/env python3


from pyrimidine import MonoIndividual, BinaryChromosome
from pyrimidine import StandardPopulation, HOFPopulation
from pyrimidine.benchmarks.optimization import Knapsack

import pytest


@pytest.fixture
def example_problem():
    n_bags = 10
    _evaluate = Knapsack.random(n_bags)
    return _evaluate


@pytest.fixture(scope='class')
def example(example_problem):
    
    _evaluate = example_problem

    class ExampleIndividual(MonoIndividual):

        element_class = BinaryChromosome.set(default_size=n_bags)

        def _fitness(self) -> float:
            return _evaluate(chromosome)

    class ExamplePopulation(StandardPopulation):
        element_class = ExampleIndividual
        default_size = 8

    return ExamplePopulation, ExampleIndividual

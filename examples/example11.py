#!/usr/bin/env python3

from pyrimidine import BinaryChromosome, BaseIndividual, AgePopulation, AgeIndividual, StandardPopulation

from pyrimidine.benchmarks.optimization import *

from digit_converter import *


def example1():
    evaluate = MLE.random(size=100)

    c = IntervalConverter(lb=0.001, ub=100)
    d = IntervalConverter(lb=-100, ub=100)


    class YourIndividual(BaseIndividual):
        element_class = BinaryChromosome

        def decode(self):
            return c(self[0]), c(self[1])

        def _fitness(self):
            return evaluate(self.decode())


    class YourPopulation(StandardPopulation):
        element_class = YourIndividual

    pop = YourPopulation.random(n_individuals=50, n_chromosomes=2, size=32)
    pop.evolve(n_iter=100, verbose=True)


def example2():
    evaluate = MixMLE.random()

    c = IntervalConverter(lb=-10, ub=10)
    d = unitIntervalConverter

    class YourIndividual(BaseIndividual):
        element_class = BinaryChromosome

        def decode(self):
            return (c(self[0]), c(self[1])), (d(self[2]), 1-d(self[2]))

        def _fitness(self):
            return evaluate(*(self.decode()))


    class YourPopulation(StandardPopulation):
        element_class = YourIndividual

    pop = YourPopulation.random(n_individuals=50, n_chromosomes=3, size=16)
    pop.evolve(n_iter=100, verbose=True)

example2()

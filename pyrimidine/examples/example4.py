#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from beagle import *
from beagle.local_search import *
from random import randint


from beagle.benchmarks.special import *

from digit_converter import *

import numpy as np

c=IntervalConverter(-30,30)

evaluate = lambda x: - rosenbrock(10)(x)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class ExampleIndividual(BaseIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome
    default_size = 10

    def _fitness(self):
        x = self.decode()
        return evaluate(x)


class MyIndividual(ExampleIndividual, SimulatedAnnealing):

    def get_neighbour(self):
        cpy = self.clone()
        r = randint(0, self.n_chromosomes-1)
        cpy.chromosomes[r].mutate()
        return cpy


if __name__ == '__main__':
    stat={'Fitness':'fitness', 'Best Fitness': lambda pop: pop.best_individual.fitness}
    
    SGAPopulation.element_class = ExampleIndividual

    ga = SGAPopulation.random(n_individuals=20, n_chromosomes=10, size=10)
    ga.mate_prob = 0.9

    d= ga.history(ngen=10, stat=stat)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')

    LocalSearchPopulation.element_class = MyIndividual

    ga = LocalSearchPopulation.random(n_individuals=20, n_chromosomes=10, size=10)
    ga.mate_prob = 0.9
    d= ga.history(ngen=10, stat=stat)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')
    ax.legend(('Traditional','Traditional best', 'SA', 'SA best'))
    plt.show()

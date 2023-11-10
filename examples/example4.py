#!/usr/bin/env python3

from random import randint
import numpy as np

from pyrimidine import *
from pyrimidine.local_search import *
from pyrimidine.benchmarks.special import *

from digit_converter import *


c=IntervalConverter(-30,30)

evaluate = lambda x: - rosenbrock(x)


class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class ExampleIndividual(PolyIndividual):
    """
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
    stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
    
    StandardPopulation.element_class = ExampleIndividual

    ga = StandardPopulation.random(n_individuals=20, size=10)
    ga.mate_prob = 0.9

    data= ga.evolve(n_iter=10, stat=stat, history=True)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)

    LocalSearchPopulation.element_class = MyIndividual

    lga = LocalSearchPopulation.random(n_individuals=20, n_chromosomes=10, size=10)
    lga.mate_prob = 0.9
    d= lga.evolve(n_iter=10, stat=stat, history=True)
    d[['Mean Fitness', 'Best Fitness']].plot(ax=ax, style='.-')
    ax.legend(('Traditional','Traditional best', 'SA', 'SA best'))
    plt.show()

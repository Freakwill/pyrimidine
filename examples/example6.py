#!/usr/bin/env python3


from pyrimidine.benchmarks.special import *
from pyrimidine import *
from digit_converter import IntervalConverter, unitIntervalConverter

import numpy as np


evaluate = lambda x: -rosenbrock(x)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return IntervalConverter(-30,30)(self)

class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)

class _Mixin:
    def _fitness(self):
        x = [i.decode() for i in self]
        return evaluate(x)

class ExampleIndividual(_Mixin, MixedIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (_Chromosome,)*20


class MyIndividual(_Mixin, SelfAdaptiveIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (_Chromosome,)*20 + (FloatChromosome,)
    ranking = None

    # @property
    # def mutate_prob(self):
    #     return self.chromosomes[-1]

    # @property
    # def mate_prob(self):
    #     return self.chromosomes[-2]

class MyPopulation(HOFPopulation):
    element_class = MyIndividual


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    stat = {'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness', 'Standard Deviation': lambda pop:np.std([ind.fitness for ind in pop])}

    MyPopulation.element_class = MyIndividual
    pop = MyPopulation.random(n_individuals=40, sizes=(8,)*20+(2,))
    cpy = pop.clone(type_=StandardPopulation[ExampleIndividual])
    pop.mate_prob = 1
    pop.mutate_prob = 1
    d = pop.evolve(n_iter=500, stat=stat, history=True, period=10)

    d[['Mean Fitness', 'Best Fitness']].plot(ax=ax, style='.-')
    d['Standard Deviation'].plot(ax=ax2, style='--')

    cpy.mate_prob = 0.9
    d = cpy.evolve(n_iter=500, stat=stat, history=True, period=10)
    d[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    d['Standard Deviation'].plot(ax=ax2, style='--')

    ax.legend(('Traditional','Traditional best', 'Self-adaptive', 'Self-adaptive best'))
    ax2.legend(('Traditional', 'New'))
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from beagle.benchmarks.special import *


from beagle import *
from digit_converter import *

import numpy as np


def evaluate(x):
    return -rosenbrock()(x)

c=IntervalConverter(-30,30)


class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)

class ExampleIndividual(BaseIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome

    def _fitness(self):
        x = [self[k].decode() for k in range(5)]
        return evaluate(x)

class MyIndividual(ExampleIndividual, MixIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (_Chromosome,)*5 + (uChromosome,)
    ranking = None
    threshold = 0.25

    @property
    def threshold(self):
        return self.chromosomes[-1].decode()


    def mate(self, other, mate_prob=None):

        if other.ranking:
            if self.threshold <= other.ranking:
                return super(MyIndividual, self).mate(other, mate_prob=1)
            else:
                mate_prob = self.threshold
                return super(MyIndividual, self).mate(other, mate_prob=mate_prob)
        else:
            return super(MyIndividual, self).mate(other)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    def transitate(self, *args, **kwargs):
        """
        Transitation of the states of population
        Standard flow of the Genetic Algorithm
        """
        self.sort()
        self.select()
        self.mate()
        self.mutate()


if __name__ == '__main__':

    SGAPopulation.element_class = ExampleIndividual

    pop = SGAPopulation.random(n_individuals=14, n_chromosomes=5, size=10)
    pop.mate_prob = 0.9
    stat = {'Fitness':'fitness', 'Best Fitness': 'best_fitness'}
    d = pop.history(ngen=100, stat=stat)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')


    pop = MyPopulation.random(n_individuals=10, sizes=[10,10,10,10,10])

    pop.mate_prob = 0.9
    d = pop.history(ngen=100, stat=stat)
    d.to_csv('h.csv')
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')
    ax.legend(('Traditional','Traditional best', 'Trait', 'Trait best'))
    plt.show()


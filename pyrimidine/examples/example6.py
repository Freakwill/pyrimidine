#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine.benchmarks.special import *


from pyrimidine import *
from digit_converter import *

import numpy as np


evaluate = lambda x: -rosenbrock(20)(x)

c=IntervalConverter(-30,30)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)

class ExampleIndividual(MixIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (_Chromosome,)*20

    def _fitness(self):
        x = [self[_].decode() for _ in range(20)]
        return evaluate(x)

class MyIndividual(ExampleIndividual, TraitThresholdIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (_Chromosome,)*20 + (uChromosome,)*3
    ranking = None
    threshold = 0.25

    @property
    def threshold(self):
        return self.chromosomes[-1].decode()

    @property
    def mutate_prob(self):
        return self.chromosomes[-2].decode()

    @property
    def mate_prob(self):
        return self.chromosomes[-3].decode()

class MyPopulation(SGAPopulation):
    def transitate(self):
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

    pop = SGAPopulation.random(n_individuals=40, sizes=(8,)*20)
    pop.mate_prob = 0.9
    stat = {'Fitness':'fitness', 'Best Fitness': lambda pop: pop.best_individual.fitness}
    d= pop.history(n_iter=350, stat=stat)
    d.to_csv('h1.csv')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')

    MyPopulation.element_class = MyIndividual
    pop = MyPopulation.random(n_individuals=40, sizes=(8,)*20+(8,)*3)

    pop.mate_prob = 1
    pop.mutate_prob = 1
    stat.update({'Best Mate_prob': lambda pop:pop.best_individual.mate_prob,
        'Best Mutate_prob': lambda pop:pop.best_individual.mutate_prob,
        'Best Threshold': lambda pop:pop.best_individual.threshold,})
    d = pop.history(ngen=350, stat=stat)
    d.to_csv('h.csv')
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')
    ax.legend(('Traditional','Traditional best', 'Trait', 'Trait best'))
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine.benchmarks.special import *


from pyrimidine import *
from digit_converter import *


ndim = 10
def evaluate(x):
    return -rosenbrock(ndim)(x)

c=IntervalConverter(-5,5)


class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)


class Mixin:
    def _fitness(self):
        x = [self[k].decode() for k in range(ndim)]
        return evaluate(x)

class ExampleIndividual(Mixin, MultiIndividual):
    element_class = _Chromosome

class MyIndividual(Mixin, MixIndividual[(_Chromosome,)*ndim + (uChromosome,)]):
    """my own individual class
    
    Method `mate` is overriden.
    """
    ranking = None
    threshold = 0.25

    @property
    def threshold(self):
        return self.chromosomes[-1].decode()


    def mate(self, other, mate_prob=None):

        if other.ranking and self.ranking:
            if self.threshold <= other.ranking:
                if other.threshold <= self.ranking:
                    return super(MyIndividual, self).mate(other, mate_prob=0.95)
                else:
                    mate_prob = 1-other.threshold
                    return super(MyIndividual, self).mate(other, mate_prob)
            else:
                if other.threshold <= self.ranking:
                    mate_prob = 1-self.threshold
                    return super(MyIndividual, self).mate(other, mate_prob=0.95)
                else:
                    mate_prob = 1-(self.threshold+other.threshold)/2
                    return super(MyIndividual, self).mate(other, mate_prob)
        else:
            return super(MyIndividual, self).mate(other)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    def transit(self, *args, **kwargs):
        self.sort()
        self.select()
        self.mate()
        self.mutate()


if __name__ == '__main__':

    stat = {'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    _Population = SGAPopulation[ExampleIndividual]
    pop = MyPopulation.random(n_individuals=20, sizes=[8]*ndim+[8])
    cpy = pop.clone(_Population)
    d = cpy.evolve(n_iter=100, stat=stat, history=True)
    ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')

    d = pop.evolve(n_iter=100, stat=stat, history=True)
    ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')
    ax.legend(('Traditional mean','Traditional best', 'New mean', 'New best'))
    plt.show()


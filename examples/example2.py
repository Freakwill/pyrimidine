#!/usr/bin/env python3

from pyrimidine.benchmarks.special import *
from pyrimidine import *

from digit_converter import *


ndim = 12
def evaluate(x):
    return -rosenbrock(x)


class _Chromosome(BinaryChromosome):
    def decode(self):
        return IntervalConverter(-1, 1)(self)


class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)


def _fitness(obj):
    return evaluate(obj.decode())

ExampleIndividual = MultiIndividual[_Chromosome].set_fitness(_fitness)

class MyIndividual(MixedIndividual[(_Chromosome,)*ndim + (uChromosome,)].set_fitness(_fitness)):
    """my own individual class
    
    Method `mate` is overriden.
    """
    ranking = None


    @property
    def threshold(self):
        return self.chromosomes[-1].decode()


    def mate(self, other, mate_prob=None):

        if other.ranking and self.ranking:
            if self.threshold <= other.ranking:
                if other.threshold <= self.ranking:
                    return super().mate(other, mate_prob=0.99)
                else:
                    mate_prob = 1 - other.threshold
                    return super().mate(other, mate_prob)
            else:
                if other.threshold <= self.ranking:
                    mate_prob = 1 - self.threshold
                    return super().mate(other, mate_prob=0.95)
                else:
                    mate_prob = 1 - (self.threshold+other.threshold)/2
                    return super().mate(other, mate_prob)
        else:
            return super().mate(other)


class MyPopulation(HOFPopulation[MyIndividual]):

    def transit(self, *args, **kwargs):
        self.sort()
        super().transit(*args, **kwargs)


if __name__ == '__main__':

    stat = {'Mean Fitness':'mean_fitness',
        'Best Fitness': 'best_fitness'}

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pop = MyPopulation.random(n_individuals=40, size=[8]*ndim+[8])
    cpy = pop.clone(HOFPopulation[ExampleIndividual])
    d = cpy.evolve(n_iter=200, stat=stat, history=True)
    ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')
    d = pop.evolve(n_iter=200, stat=stat, history=True)
    ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')
    ax.legend(('Traditional mean', f'Traditional best({cpy.best_fitness})', 'New mean', f'New best({pop.best_fitness})'))
    ax.set_title('Comparison of two GAs')
    plt.show()


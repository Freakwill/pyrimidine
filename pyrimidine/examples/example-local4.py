#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.matrix import *

from digit_converter import *


N, p = 200, 50
c = 3
evaluate = NMF.random(N=N, p=p)


class _Individual(MultiIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """

    element_class = ProbabilityChromosome

    def _fitness(self):
        A = np.vstack(self.chromosomes[:N])
        B = np.vstack(self.chromosomes[N:]).T
        return evaluate(A, B)


class MyIndividual(_Individual, SimulatedAnnealing):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        r = randint(0, self.n_chromosomes-1)
        cpy.chromosomes[r].mutate()
        return cpy

    # def get_neighbour(self):
    #     # select a neighour randomly
    #     cpy = self.clone(fitness=None)
    #     cpy.chromosomes = [chromosome.random_neighbour() for chromosome in self.chromosomes]
    #     return cpy


i = MyIndividual.random(size=20, sizes=(3,)* N + (p,)*c)

print(i, i.fitness)

data = i.history()
print(i, i.fitness)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data['Fitness'])

plt.show()

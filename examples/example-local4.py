#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.matrix import NMF as NMF_

from digit_converter import *


N, p = 50, 10
c = 3
evaluate = NMF_.random(N=N, p=p)

class _PChromosome(ProbabilityChromosome):

    def random_neighbour(self):
        # select a neighour randomly
        r = self.random(size=self.n_genes)
        epsilon = 0.001
        return self + epsilon * r

class _Chromosome(FloatChromosome):

    def random_neighbour(self):
        # select a neighour randomly
        r = self.random(size=self.n_genes)
        epsilon = 0.001
        return self + epsilon * r


class _Individual(MixedIndividual, SimulatedAnnealing):
    """base class of individual

    You should implement the methods, cross, mute
    """

    element_class = (FloatChromosome,) * N + (ProbabilityChromosome,) * p

    def _fitness(self):
        A = np.vstack(self.chromosomes[:N])
        B = np.vstack(self.chromosomes[N:])
        return evaluate(A, B.T)


class YourIndividual(_Individual):
    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.mutate()
        return cpy


class MyIndividual(_Individual):
    element_class = (_Chromosome,) * N + (_PChromosome,) * p

    def get_neighbour(self):
        # select a neighour randomly
        cpy = self.clone(fitness=None)
        cpy.chromosomes = [chromosome.random_neighbour() for chromosome in self.chromosomes]
        return cpy


from sklearn.decomposition import NMF
nmf = NMF(n_components=c)
W = nmf.fit_transform(evaluate.M)
H = nmf.components_
err = -evaluate(W, H)

i = MyIndividual.random(sizes=(c,)* (N + p))
j = i.clone()
data = i.evolve(stat={'Error': lambda i: -i.fitness}, n_iter=100, history=True)
yourdata = j.evolve(stat={'Error': lambda i: -i.fitness}, n_iter=100, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(101), yourdata['Error'], 'bo', np.arange(101), data['Error'], 'r+', [0, 100], [err, err], 'k--')
ax.legend(('My Error', 'Your Error', 'EM Error'))
plt.show()

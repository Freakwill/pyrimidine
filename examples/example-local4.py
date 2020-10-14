#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.matrix import NMF as NMF_

from digit_converter import *


N, p = 50, 10
c = 2
evaluate = NMF_.random(N=N, p=p)

class _Chromosome(ProbabilityChromosome):

    def random_neighbour(self):
        # select a neighour randomly
        r = self.random(size=self.n_genes)
        epsilon = 0.01
        return self + epsilon * (r - self)


class _Individual(MultiIndividual, SimulatedAnnealing):
    """base class of individual

    You should implement the methods, cross, mute
    """

    def _fitness(self):
        A = np.vstack(self.chromosomes[:N])
        B = np.vstack(self.chromosomes[N:]).T
        return evaluate(A, B)


class YourIndividual(_Individual):
    element_class = ProbabilityChromosome
    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.mutate()
        return cpy


class MyIndividual(_Individual):
    element_class = _Chromosome

    def get_neighbour(self):
        # select a neighour randomly
        cpy = self.clone(fitness=None)
        cpy.chromosomes = [chromosome.random_neighbour() for chromosome in self.chromosomes]
        return cpy


from sklearn.decomposition import NMF
nmf = NMF(n_components=c)
W = nmf.fit_transform(evaluate.M)
H = nmf.components_
err = -evaluate(W, H.T)

i = MyIndividual.random(sizes=(c,)* N + (p,)*c)
j = i.clone()
data = i.get_history(stat={'Error': lambda i: -i.fitness}, n_iter=200)
yourdata = j.get_history(stat={'Error': lambda i: -i.fitness}, n_iter=200)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(200), yourdata['Error'], 'bo', np.arange(200), data['Error'], 'r+', [0, 200], [err, err], 'k--')
ax.legend(('My Error', 'Your Error', 'EM Error'))
plt.show()

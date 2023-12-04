#!/usr/bin/env python3


import numpy as np

from pyrimidine import ProbabilityChromosome, FloatChromosome, MixedIndividual, HOFPopulation

from pyrimidine.benchmarks.matrix import NMF as NMF_


N, p = 50, 10
c = 3
evaluate = NMF_.random(N=N, p=p) # (A, B) --> |C-AB|/|C|

class _PChromosome(ProbabilityChromosome):
    default_size = c

    def random_neighbour(self):
        # select a neighour randomly
        r = self.random()
        epsilon = 0.001
        return self + epsilon * r

class _Chromosome(FloatChromosome):
    default_size = c
    def random_neighbour(self):
        # select a neighour randomly
        r = self.random()
        epsilon = 0.001
        return self + epsilon * r


class _Individual(MixedIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """

    element_class = (FloatChromosome,) * N + (ProbabilityChromosome,) * p

    def _fitness(self):
        A = np.vstack(self.chromosomes[:N])
        B = np.vstack(self.chromosomes[N:])
        f = evaluate(A, B.T)
        return f


class YourIndividual(_Individual):
    pass


class MyIndividual(_Individual):
    element_class = (_Chromosome,) * N + (_PChromosome,) * p


YourPopulation = HOFPopulation[YourIndividual].set(default_size=15)
MyPopulation = HOFPopulation[MyIndividual].set(default_size=15)


pop = MyPopulation.random()
pop2 = pop.copy(type_=YourPopulation)
data = pop.evolve(stat={'Error': lambda pop: - pop.best_fitness}, n_iter=250, history=True, period=5)
yourdata = pop2.evolve(stat={'Error': lambda pop: - pop.best_fitness}, n_iter=250, history=True, period=5)


from sklearn.decomposition import NMF
nmf = NMF(n_components=c)
W = nmf.fit_transform(evaluate.M)
H = nmf.components_
err = - evaluate(W, H)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
L = len(yourdata['Error'])
ax.plot(np.arange(L), yourdata['Error'], 'b', np.arange(L), data['Error'], 'r', [0, L], [err, err], 'k--')
ax.legend(('My Error', 'Your Error', 'EM Error'))
ax.set_xlabel('Generations * 5')
ax.set_ylabel('Error')
ax.set_title('solve NMF by GA')
plt.show()

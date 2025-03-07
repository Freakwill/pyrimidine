#!/usr/bin/env python3

"""
Test for Probabilitical GA
"""

import numpy as np

# from pyrimidine.deco import fitness_cache

from pyrimidine import (MixedIndividual, ProbabilityChromosome, FloatChromosome,
    StandardPopulation, MultiPopulation)
from pyrimidine.benchmarks.matrix import NMF

N=500
p=100
nmf = NMF.random(N, p)

from sklearn.decomposition import NMF as skNMF
sknmf = skNMF(n_components=3)
W = sknmf.fit_transform(nmf.M)
H = sknmf.components_

# HN = H.sum(axis=1)[:,None]
# H /= HN;
# WN = W.sum(axis=1)
# W /= WN;
# i = _Individual([ProbabilityChromosome(w) for w in W] + [ProbabilityChromosome(h) for h in H] + FloatChromosome(HN * WN))

e = -nmf(W, H)


# @fitness_cache
class _Individual(MixedIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = (ProbabilityChromosome // N, ProbabilityChromosome // N, ProbabilityChromosome // N,
        ProbabilityChromosome // p, ProbabilityChromosome // p, ProbabilityChromosome // p,
        FloatChromosome // 3)
    default_size = 7

    def decode(self):
        A = np.column_stack((self[0], self[1], self[2]))
        B = np.column_stack((self[3], self[4], self[5]))
        return A, B.T, self[6]

    def _fitness(self):
        """
        select ni from n, ti from t
        the sum of ni ~ 10, while ti is repeated rarely
        """
        return nmf(*self.decode())


if __name__ == '__main__':
    StandardPopulation.element_class = _Individual
    _Population = StandardPopulation[_Individual] // 50
    pop = StandardPopulation.random()

    _MultiPopulation = MultiPopulation[_Population] // 4
    mpop = _MultiPopulation.random()

    stat={'Mean Error':lambda pop : -pop.mean_fitness, 'Best Error':lambda pop : -pop.max_fitness, 'STD Error': lambda pop:np.std(pop.all_fitness)}

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111); ax2 = ax.twinx()

    data = pop.evolve(stat=stat, max_iter=100, history=True)
    data[['Mean Error', 'Best Error']].plot(ax=ax)
    data[['STD Error']].plot(ax=ax2, style='--')

    data = mpop.evolve(stat=stat, max_iter=100, history=True)
    data[['Mean Error', 'Best Error']].plot(ax=ax)
    data[['STD Error']].plot(ax=ax2, style='--')

    ax.plot([0,100], [e,e], '-k')

    ax.set_xlabel('Generations')
    ax.set_ylabel('Error')
    plt.show()


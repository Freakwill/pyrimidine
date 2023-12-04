#!/usr/bin/env python3

"""
function approximation by GA
"""

import numpy as np

from pyrimidine import makeIndividual, HOFPopulation, FloatChromosome
from pyrimidine.benchmarks.approximation import Function1DApproximation, _basis, lin_comb

from pyrimidine.deco import fitness_cache


evaluate = Function1DApproximation(function=lambda x:10*np.arctan(x), lb=-2, ub=2, basis=_basis)
n_basis = len(evaluate.basis)


@fitness_cache
class MyIndividual(makeIndividual(FloatChromosome, n_chromosomes=1, size=n_basis)):
    def _fitness(self):
        return evaluate(self.chromosome)


MyPopulation = HOFPopulation[MyIndividual]

pop = MyPopulation.random(n_individuals=100)

stat = {'Best Fitness': 'best_fitness', 'Mean Fitness': 'mean_fitness'}
data = pop.evolve(n_iter=200, stat=stat, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax1, ax2 = fig.subplots(1,2)
x = evaluate.x
y = evaluate.y
y_ = lin_comb(x, pop.best_individual.chromosome, evaluate.basis)
ax1.plot(x, y, x, y_)
ax1.legend(('Original Function', 'Approximating Function'))
data[['Best Fitness', 'Mean Fitness']].plot(ax=ax2)
ax2.set_xlabel('Generations')
ax2.set_ylabel('Fitness')
plt.show()

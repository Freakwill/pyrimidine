#!/usr/bin/env python3

"""
function approximation by GA
"""

from pyrimidine import *
from pyrimidine.benchmarks.approximation import Function1DApproximation, lin_comb, _my_basis, _tri_basis
import numpy as np

evaluate = Function1DApproximation(function=lambda x:10*np.arctan(x), lb=-2, ub=2, basis=_my_basis)
n_basis = len(evaluate.basis)

class MyIndividual(makeIndividual(FloatChromosome, n_chromosomes=1, size=n_basis)):
    def _fitness(self):
        return evaluate(self.chromosome)


class MyPopulation(HOFPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=200)

stat={'Best Fitness': 'best_fitness', 'Mean Fitness': 'mean_fitness'}
data=pop.evolve(n_iter=500, stat=stat, history=True)


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.approximation import _basis, n_basis_, Function1DApproximation
import numpy as np

evaluate = Function1DApproximation(function=lambda x:10*np.arctan(x),lb=-1, ub=1)

class MyIndividual(MonoFloatIndividual):
    def _fitness(self):
        return evaluate(self.chromosome)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=200, size=n_basis_)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
pop.evolve(n_iter=250, verbose=False)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
x = evaluate.x
y = evaluate.y
coefs = pop.best_individual.chromosome
yy = np.sum([c*b(x) for c, b in zip(coefs, _basis)], axis=0)
ax.plot(x, y, x, yy)
ax.legend(('Original Function', 'Approximating Function'))
plt.show()

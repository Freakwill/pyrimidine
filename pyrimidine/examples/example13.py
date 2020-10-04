#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

X = np.linspace(-3, 3, 100)
y = np.arctan(X)

evaluate = Fitting(X, y)

class Gene1(FloatGene):
    lb, ub = -1, 1

class Chromosome1(FloatChromosome):
    element_class = Gene1

class Gene2(FloatGene):
    lb, ub = -3, 3

class Chromosome2(FloatChromosome):
    element_class = Gene2

class MyIndividual(MixIndividual):
    element_class = Chromosome1, Chromosome2, Chromosome2


    def _fitness(self):
        return evaluate(*self.chromosomes)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=30, n_chromosomes=2, sizes=(10, 10, 10))

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
params = pop.best_individual.chromosomes
yy = evaluate.fit(*params)
pop.evolve(n_iter=200, verbose=False)
params = pop.best_individual.chromosomes
yyy = evaluate.fit(*params)
ax.plot(X, y, X, yy, X, yyy)
ax.legend(('Original Function', 'Approximating Function (G1)', 'Approximating Function (G100)'))
plt.show()

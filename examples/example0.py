#!/usr/bin/env python3

"""
Test for methods/operators: A[B], set, set_fitness, //
"""

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)
MonoBinaryIndividual = MonoIndividual[BinaryChromosome.set(default_size=n)].set_fitness(lambda o: _evaluate(o.chromosome))
MyPopulation = StandardPopulation[MonoBinaryIndividual] // 10

pop = MyPopulation.random()

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

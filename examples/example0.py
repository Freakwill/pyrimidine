#!/usr/bin/env python3

"""
Test for methods/operators: A[B], set, set_fitness, //
"""

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *

n_bags = 100
_evaluate = Knapsack.random(n_bags)
def _fitness(o):
    return _evaluate(o)
_Individual = (BinaryChromosome // n_bags).set_fitness(_fitness)
MyPopulation = StandardPopulation[_Individual] // 5

pop = MyPopulation.random()

# import concurrent.futures
# MyPopulation.map = concurrent.futures.ProcessPoolExecutor().map

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'max_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Have a test')
plt.show()

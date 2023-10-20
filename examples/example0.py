#!/usr/bin/env python3

from pyrimidine import MonoIndividual, BinaryChromosome, HOFPopulation
from pyrimidine.benchmarks.optimization import *

n_bags = 50
_evaluate = Knapsack.random(n_bags)

# Define individual
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome.set(default_size=n_bags)
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)

# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

# Define Population
class MyPopulation(HOFPopulation):
    element_class = MyIndividual
    default_size = 10

pop = MyPopulation.random()

stat = {
    'Mean Fitness':'fitness', 'Best Fitness':'best_fitness',
    'Standard Deviation of Fitnesses': 'std_fitness', 'number': lambda o: len(o.individuals)
    }
data = pop.evolve(stat=stat, n_iter=100, history=True, verbose=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(loc='upper left')
data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='y-.')
ax2.legend(loc='lower right')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

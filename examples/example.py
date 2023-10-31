#!/usr/bin/env python3

"""
An ordinary example of the usage of `pyrimidine`
"""

from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
from pyrimidine.benchmarks.optimization import *

n_bags = 50
_evaluate = Knapsack.random(n_bags)  # : 0-1 array -> float

# Define the individual class
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome.set(default_size=n_bags)
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)

""" Equiv. to
    MyIndividual = MonoIndividual[BinaryChromosome.set(default_size=n_bags)].set_fitness(_evaluate)
"""

# Define the population class
class MyPopulation(StandardPopulation):
    element_class = MyIndividual
    default_size = 8

""" Equiv. to
    MyPopulation = StandardPopulation[MyIndividual].set(default_size=8)
    or, as a population of chromosomes
    MyPopulation = StandardPopulation[BinaryChromosome.set(default_size=n_bags).set_fitness(_evaluate)].set(default_size=8)
"""

pop = MyPopulation.random()

# Define statistics of population
stat = {
    'Mean Fitness': 'mean_fitness',
    'Best Fitness': 'best_fitness',
    'Standard Deviation of Fitnesses': 'std_fitness',
    'number': lambda pop: len(pop.individuals)  # or `'n_individuals'`
    }

# Do statistical task and print the results through the evoluation
data = pop.evolve(stat=stat, n_iter=200, history=True, verbose=False)

# Visualize the data
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
ax.set_title('Demo of solving the knapsack problem by GA')
plt.show()

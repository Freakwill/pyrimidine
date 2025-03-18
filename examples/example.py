#!/usr/bin/env python3

"""
An ordinary example of the usage of `pyrimidine`
"""

from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import fitness_cache


n_bags = 50
_evaluate = Knapsack.random(n_bags)  # : 0-1 array -> float

print(_evaluate.w)

# Define the individual class
@fitness_cache
class MyIndividual(MonoIndividual):

    element_class = BinaryChromosome.set(default_size=n_bags)
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)

# Equiv. to
# MyIndividual = (BinaryChromosome // n_bags).set_fitness(_evaluate) @ fitness_cache


# Define the population class
class MyPopulation(StandardPopulation):
    element_class = MyIndividual
    default_size = 20

""" Equiv. to
    MyPopulation = StandardPopulation[MyIndividual] // 20
    or, as a population of chromosomes
    MyPopulation = StandardPopulation[(BinaryChromosome // n_bags).set_fitness(_evaluate)] // 8
"""

pop = MyPopulation.random()


if __name__ == '__main__':

    # Define statistics of population
    stat = {
        'Mean Fitness': 'mean_fitness',
        'Max Fitness': 'max_fitness',
        'Standard Deviation of Fitnesses': 'std_fitness',
        # 'number': lambda pop: len(pop.individuals)  # or `'n_individuals'`
        }

    # Do statistical task and print the results through the evoluation
    data = pop.evolve(stat=stat, max_iter=100, history=True)

    # Visualize the data
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data[['Mean Fitness', 'Max Fitness']].plot(ax=ax)
    ax.legend(loc='upper left')
    data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='y-.')
    ax2.legend(loc='lower right')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.set_title('Demo of solving the knapsack problem by GA')
    plt.show()

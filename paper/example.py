#!/usr/bin/env python3

"""
An ordinary example of the usage of `pyrimidine`
"""

from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import fitness_cache

n_bags = 50
_evaluate = Knapsack.random(n_bags)  # : 0-1 array -> float

# Define the individual class
# @fitness_cache
# class MyIndividual(MonoIndividual):

#     element_class = BinaryChromosome.set(default_size=n_bags)
#     def _fitness(self) -> float:
#         # To evaluate an individual!
#         return _evaluate(self.chromosome)

# Equiv. to
MyIndividual = (BinaryChromosome // n_bags).set_fitness(_evaluate)


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
        'Max Fitness': 'max_fitness',
        'Mean Fitness': 'mean_fitness',
        'Standard Deviation of Fitnesses': 'std_fitness',
        # '0.85-Quantile': lambda pop: np.quantile(pop.get_all_fitness(), 0.85),
        # '0.15-Quantile': lambda pop: np.quantile(pop.get_all_fitness(), 0.15),
        # 'Median Fitness': lambda pop: np.median(pop.get_all_fitness())
        }

    # Do statistical task and print the results through the evoluation
    data = pop.evolve(stat=stat, max_iter=100, verbose=True)

    # Visualize the data
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # data[['Best Fitness', 'Mean Fitness']].plot(ax=ax)
    # ax.legend(loc='upper left')
    # std = data['Standard Deviation of Fitnesses']
    # ax.set_xlabel('Generations')
    # ax.set_ylabel('Fitness')

    # y = data['Mean Fitness']
    # ub = y + std * 0.5
    # lb = y - std * 0.5
    # ax.fill_between(np.arange(101), lb, ub,
    # alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    # plt.savefig('plot-history.png')

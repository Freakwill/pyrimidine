#!/usr/bin/env python3


from pyrimidine import binaryIndividual
from pyrimidine.population import *

from pyrimidine.benchmarks.optimization import *

n_bags = 50
_evaluate = Knapsack.random(n_bags)

class MyIndividual(binaryIndividual(n_bags)):
    def _fitness(self):
        return _evaluate(self.chromosome)


class _Population1(StandardPopulation):
    element_class = MyIndividual
    default_size = 8

    def transition(self, *args, **kwargs):
        super().transition(*args, **kwargs)
        print(self.get_all_fitness(), self.fitness)


# class _Population2(HOFPopulation):
#     element_class = MyIndividual
#     default_size = 10

pop1 = _Population1.random(size=n_bags)
# build population 2 with the same initial values to population 1, by `copy` method
# pop2 = pop1.copy(type_=_Population2)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'max_fitness'}
data1 = pop1.evolve(stat=stat, n_iter=50, history=True)
# data2 = pop2.evolve(stat=stat, n_iter=100, history=True)


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
# # data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
# ax.legend(('Mean', 'Best', 'Mean(HOF)', 'Best(HOF)'))
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
# plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""An easy knapsack problem

Your first example of pyrimidine
"""

from pyrimidine import MonoBinaryIndividual, BaseEnvironment
from pyrimidine.population import StandardPopulation, HOFPopulation

from pyrimidine.benchmarks.optimization import Knapsack

n_bags = 50
class Env(BaseEnvironment):
    _evaluate = Knapsack.random(n_bags)

with Env() as env:
    _Individual = MonoBinaryIndividual.set_fitness(lambda o: env._evaluate(o.chromosome))


    class _Population1(StandardPopulation):
        element_class = _Individual
        default_size = n_bags

    class _Population2(HOFPopulation):
        element_class = _Individual
        default_size = n_bags


    pop1 = _Population1.random(size=n_bags)
    pop2 = pop1.clone(type_=_Population2)

    stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
    data1 = pop1.evolve(stat=stat, n_iter=100, history=True)
    data2 = pop2.evolve(stat=stat, n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('M1', 'B1', 'M2', 'B2'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Comparison between Standard GA and HOF GA')
plt.show()

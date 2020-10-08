#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.optimization import *

from digit_converter import *


evaluate = Knapsack.random()



class _Individual(MonoBinaryIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """

    def _fitness(self):
        return evaluate(self)


class MyIndividual(_Individual, SimulatedAnnealing):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        r = randint(0, self.n_chromosomes-1)
        cpy.chromosomes[r].mutate()
        return cpy


i = MyIndividual.random(size=20)

print(i, i.fitness)

data = i.history(stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'})
print(i, i.fitness)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'], 'o-')
ax.legend(('Mean Fitness', 'Best Fitness'))
plt.show()



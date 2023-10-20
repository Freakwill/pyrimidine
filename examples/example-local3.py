#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.local_search import SimulatedAnnealing

from pyrimidine.benchmarks.optimization import *


evaluate = Knapsack.random()


class _Individual(MonoIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = BinaryChromosome

    def _fitness(self):
        return evaluate(self.chromosome)


class MyIndividual(SimulatedAnnealing, _Individual):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.chromosome.mutate()
        return cpy


i = MyIndividual.random(size=20)

data = i.evolve(n_iter=100, history=True)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data.plot(ax=ax)
ax.set_title('Simulated Annealing')
ax.set_ylabel('Value of Objective')
ax.set_xlabel('Iterations')
plt.show()


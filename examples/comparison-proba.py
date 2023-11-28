#!/usr/bin/env python3

import copy

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
n_bags = 50
evaluate = Knapsack.random(n=n_bags)


class YourIndividual(BinaryChromosome // n_bags):

    def _fitness(self):
        return evaluate(self.decode())


class YourPopulation(HOFPopulation):
    element_class = YourIndividual
    default_size = 20


from pyrimidine.deco import add_memory

@add_memory({'measure_result': None, 'fitness': None})
class MyIndividual(QuantumChromosome // n_bags):

    def _fitness(self):
        return evaluate(self.decode())

    def backup(self, check=False):
        f = super().fitness
        if not check or (self.memory['fitness'] is None or f > self.memory['fitness']):
            self._memory = {
            'measure_result': self.measure_result,
            'fitness': f
            }


class MyPopulation(HOFPopulation):

    element_class = MyIndividual
    default_size = 20

    def init(self):
        for i in self:
            i.backup()
        super().init()

    def backup(self, check=True):
        for i in self:
            i.backup(check=check)

    def transition(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """
        super().transition(*args, **kwargs)
        self.backup()


stat={'Mean Fitness': 'mean_fitness', 'Best Fitness': 'best_fitness'}
mypop = MyPopulation.random()

yourpop = YourPopulation([YourIndividual(i.decode()) for i in mypop])
mydata = mypop.evolve(n_iter=100, stat=stat, history=True)
yourdata = yourpop.evolve(n_iter=100, stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
yourdata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
mydata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'Mean Fitness(Quantum)', 'Best Fitness(Quantum)'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title(f'Demo of (Quantum)GA: {n_bags}-Knapsack Problem')
plt.show()

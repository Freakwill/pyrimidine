#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import add_memory, fitness_cache

# generate a knapsack problem randomly
n_bags = 50
evaluate = Knapsack.random(n=n_bags)

@fitness_cache
class YourIndividual(BinaryChromosome // n_bags):

    def _fitness(self):
        return evaluate(self.decode())


YourPopulation = HOFPopulation[YourIndividual] // 20


@add_memory({'solution': None, 'fitness': None})
class MyIndividual(YourIndividual):

    def backup(self, check=False):
        f = self._fitness()
        if not check or (self.memory['fitness'] is None or f > self.memory['fitness']):
            self._memory = {
            'solution': self.clone(),
            'fitness': f
            }

    @property
    def solution(self):
        if self._memory['solution'] is not None:
            return self._memory['solution']
        else:
            return self.solution


class MyPopulation(HOFPopulation):

    element_class = MyIndividual
    default_size = 20

    def init(self):
        self.backup()
        super().init()

    def backup(self, check=True):
        for i in self:
            i.backup(check=check)

    def transition(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """
        self.backup()
        super().transition(*args, **kwargs)


stat = {'Mean Fitness': 'mean_fitness', 'Best Fitness': 'best_fitness'}
mypop = MyPopulation.random()

yourpop = mypop.clone(type_=YourPopulation)
mydata = mypop.evolve(n_iter=200, stat=stat, history=True)
yourdata = yourpop.evolve(n_iter=200, stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
yourdata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
mydata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'Mean Fitness(Memory)', 'Best Fitness(Memory)'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title(f'Demo of GA: {n_bags}-Knapsack Problem')
plt.show()

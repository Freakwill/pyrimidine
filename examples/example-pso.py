#!/usr/bin/env python3

from pyrimidine.base import *
from pyrimidine.pso import Particle, ParticleSwarm

from pyrimidine.benchmarks.special import *

# generate a knapsack problem randomly
def evaluate(x):
    return -rosenbrock(8)(x)

class _Particle(Particle):
    default_size = 8
    def _fitness(self):
        return evaluate(self.position)


class MyParticleSwarm(ParticleSwarm, BasePopulation):
    element_class = _Particle
    default_size = 20

pop = MyParticleSwarm.random()


stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
print(pop.best_fitness)
data = pop.evolve(stat=stat, n_iter=100, history=True)
print(pop.best_fitness)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
axt = ax.twinx()
data[['Best Fitness']].plot(ax=ax)
data[['Mean Fitness']].plot(ax=axt, color='orange', linestyle='--')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of PSO')
plt.show()

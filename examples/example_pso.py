#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine.base import BasePopulation
from pyrimidine.pso import Particle, ParticleSwarm

from pyrimidine.benchmarks.special import *

# generate a knapsack problem randomly
def evaluate(x):
    return -rosenbrock(8)(x)

class _Particle(Particle):
    default_size = 8
    def _fitness(self):
        return evaluate(self.chromosomes[0])


class MyParticleSwarm(ParticleSwarm, BasePopulation):
    element_class = _Particle
    default_size = 20

pop = MyParticleSwarm.random()


stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=150, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

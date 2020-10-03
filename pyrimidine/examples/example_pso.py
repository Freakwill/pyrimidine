#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine.pso import Particle, ParticleSwarm

from pyrimidine.benchmarks.special import *

# generate a knapsack problem randomly
def evaluate(x):
    return -rosenbrock(8)(x)

class _Particle(Particle):
    default_size = 8
    def _fitness(self):
        return evaluate(self.chromosomes[0])


class MyParticleSwarm(ParticleSwarm):
    element_class = _Particle
    default_size = 20

pop = MyParticleSwarm.random()


stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'}
data = pop.history(stat=stat, ngen=100)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])
ax.legend(('Fitness', 'Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

#!/usr/bin/env python3


import numpy as np

from pyrimidine import *
from pyrimidine.pso import Particle, ParticleSwarm

from pyrimidine.benchmarks.special import *

# generate a knapsack problem randomly
n = 12
def evaluate(x):
    return -rosenbrock(x)


class MyParticleSwarm(ParticleSwarm):
    element_class = Particle[FloatChromosome//n].set_fitness(lambda o:evaluate(o.position))
    default_size = 20


class MyPopulation(StandardPopulation):
    element_class = (FloatChromosome // n).set_fitness(evaluate)
    default_size = 20


pop = MyParticleSwarm.random()
pop2 = MyPopulation([i[0].copy() for i in pop])


stat={'Best Fitness(PSO)': 'max_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)

stat={'Best Fitness(GA)': 'max_fitness'}
data2 = pop2.evolve(stat=stat, n_iter=100, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Best Fitness(PSO)']].plot(ax=ax)
data2[['Best Fitness(GA)']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Comparison of PSO and GA')
plt.show()

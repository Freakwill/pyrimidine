#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.pso import Particle, ParticleSwarm

from pyrimidine.benchmarks.special import *
import numpy as np

# generate a knapsack problem randomly
n = 12
def evaluate(x):
    return -rosenbrock(n)(x)

class _Particle(Particle):
    default_size = n
    def _fitness(self):
        return evaluate(self.position)


class MyParticleSwarm(ParticleSwarm, BasePopulation):
    element_class = _Particle
    default_size = 20

MyPopulation = HOFPopulation[MonoIndividual[FloatChromosome].set_fitness(lambda o: evaluate(o.chromosome)) // n] // 20

pop = MyParticleSwarm.random()
pop2 = pop.clone(type_=MyPopulation)

stat={'Best Fitness(PSO)': 'best_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)

stat={'Best Fitness(GA)': 'best_fitness'}
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

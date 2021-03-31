#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
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

MyPopulation = HOFPopulation[MonoIndividual[FloatChromosome].set_fitness(lambda o:evaluate(o.chromosome))//8] // 20


pop = MyParticleSwarm.random()

pop2 = pop.clone(type_=MyPopulation)


stat={'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)

stat={'Best Fitness2':'best_fitness'}
data2 = pop2.evolve(stat=stat, n_iter=100, history=True)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Best Fitness']].plot(ax=ax)
data2[['Best Fitness2']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

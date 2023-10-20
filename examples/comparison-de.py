#!/usr/bin/env python3


from pyrimidine.individual import MonoIndividual
from pyrimidine.population import HOFPopulation, BasePopulation
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.de import *

from pyrimidine.benchmarks.optimization import *

from pyrimidine.benchmarks.special import *

n = 10
f = rosenbrock(n=n)

class MyIndividual(MonoIndividual):
    element_class = FloatChromosome.set(default_size=n)

    def _fitness(self):
        return -f(self.chromosome)


class _Population1(DifferentialEvolution, BasePopulation):
    element_class = MyIndividual
    default_size = 10

class _Population2(HOFPopulation):
    element_class = MyIndividual
    default_size = 10

pop1 = _Population1.random()
pop2 = pop1.clone(type_=_Population2) # population 2 with the same initial values to population 1

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data1 = pop1.evolve(stat=stat, n_iter=100, history=True)
data2 = pop2.evolve(stat=stat, n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean DE', 'Best DE', 'Mean GA', 'Best GA'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Differential Evolution')
plt.show()

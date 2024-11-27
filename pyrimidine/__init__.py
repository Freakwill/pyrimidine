#!/usr/bin/env python3


from .base import *
from .mixin import *
from .errors import *
from .gene import *
from .chromosome import *
from .individual import *
from .population import *
from .multipopulation import *
from .saga import *
from .studga import *
from .pso import *
from .es import *
from .ep import *
from .de import *


__version__ = "1.6.1"

__template__ = """
from pyrimidine.chromosome import BinaryChromosome
from pyrimidine.individual import MonoIndividual, binaryIndividual
from pyrimidine.population import StandardPopulation

from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)


# Define individual
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome//n

    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)

# Equiv. to
# MyIndividual = MyIndividual[BinaryChromosome//n].set_fitness(lambda o: _evaluate(o.chromosome))
# MyIndividual = (BinaryChromosome//n).set_fitness(_evaluate) # Algebraically equiv.

# Define Population
class MyPopulation(HOFPopulation):
    element_class = MyIndividual
    default_size = 20

# Equiv. to
# MyPopulation = StandardPopulation[MyIndividual] // 20

# Define MyPopulation as a container of BinaryChromosome
# MyPopulation = StandardPopulation[BinaryChromosome//n].set_fitness(_evaluate) // 20

pop = MyPopulation.random(size=n)

stat={'Mean Fitness': 'fitness', 'Best Fitness': 'max_fitness',
  'Standard Deviation of Fitnesses': 'std_fitness'}
data = pop.evolve(stat=stat, max_iter=200, history=True)
# data = pop.ezolve(max_iter=200) # for eaziness

# Visualization
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(loc='upper left')
data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='r--')
ax2.legend(loc='lower right')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
"""

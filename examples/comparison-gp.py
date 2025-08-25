#!/usr/bin/env python3

# did not work currently.

import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def _evaluate(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    points=[x/10. for x in range(-10,10)]
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points)


toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import fitness_cache


# Define the individual class
@fitness_cache
class MyIndividual(creator.Individual):

    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)

    def mate(self, other):
        return gp.cxOnePoint(self, other)

    def mutate(self):
        gp.mutUniform(self, expr=toolbox.expr_mut, pset=pset)



# Define the population class
class MyPopulation(StandardPopulation):
    element_class = MyIndividual
    default_size = 20

    def init(self):
        return toolbox.population(self.default_size)

""" Equiv. to
    MyPopulation = StandardPopulation[MyIndividual] // 20
    or, as a population of chromosomes
    MyPopulation = StandardPopulation[(BinaryChromosome // n_bags).set_fitness(_evaluate)] // 8
"""

pop = MyPopulation.init()


if __name__ == '__main__':

    # Define statistics of population
    stat = {
        'Mean Fitness': 'mean_fitness',
        'Max Fitness': 'max_fitness',
        'Standard Deviation of Fitnesses': 'std_fitness',
        # 'number': lambda pop: len(pop.individuals)  # or `'n_individuals'`
        }

    # Do statistical task and print the results through the evoluation
    data = pop.evolve(stat=stat, max_iter=10, history=True)

    # Visualize the data
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data[['Mean Fitness', 'Max Fitness']].plot(ax=ax)
    ax.legend(loc='upper left')
    data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='y-.')
    ax2.legend(loc='lower right')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.set_title('Demo of solving the knapsack problem by GA')
    plt.show()

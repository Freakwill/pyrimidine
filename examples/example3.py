#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from pyrimidine import *
from digit_converter import *
from pyrimidine.benchmarks.neural_network import MLP

import numpy as np


N, p = 300, 3

evaluate = MLP.random(N=N, p=p)

h = 10

c = IntervalConverter(-5,5)


class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class ExampleIndividual(MixIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = FloatChromosome, FloatChromosome, FloatChromosome, _Chromosome

    def decode(self):
        return self[0].reshape(p, h), self[1], self[2], self[3].decode()

    def _fitness(self):
        return evaluate(self.decode())

class ExampleIndividual2(ExampleIndividual, TraitThresholdIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = FloatChromosome, FloatChromosome, FloatChromosome, _Chromosome, FloatChromosome

if __name__ == '__main__':

    SGAPopulation.element_class = ExampleIndividual2
    pop = SGAPopulation.random(n_individuals=50, sizes=[h*p, h, h, 8, 3])
    pop1 = pop.clone(type_=SGAPopulation)

    pop.mate_prob = pop.mutate_prob= 1
    pop1.mate_prob = 0.5
    pop1.mutate_prob = 0.9
    data = pop.history(n_iter=300, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness', 
        'mean threshold': lambda pop: np.mean([ind.threshold for ind in pop.individuals]),
        'mean mate_prob': lambda pop: np.mean([ind.mate_prob for ind in pop.individuals]),
        'mean mutate_prob': lambda pop: np.mean([ind.mutate_prob for ind in pop.individuals]),
        'best threshold': lambda pop: pop.best_.threshold,
        'best mate_prob': lambda pop: pop.best_.mate_prob,
        'best mutate_prob': lambda pop: pop.best_.mutate_prob,
        'worst desire': lambda pop: pop.worst_.desire,
        'best desire': lambda pop: pop.best_.desire
        })
    data.to_csv('h.csv')

    SGAPopulation.element_class = ExampleIndividual

    pop1.mate_prob = 0.1
    pop1.mutate_prob = 0.8
    data1 = pop1.get_history(n_iter=300)


    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax, style='--')
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)

    ax.legend(('Traditional','Traditional best', 'New', 'New best'))
    ax.set_xlabel('Generation')
    plt.show()


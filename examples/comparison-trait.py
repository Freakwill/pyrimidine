#!/usr/bin/env python3


from pyrimidine import *
from digit_converter import *
from pyrimidine.benchmarks.neural_network import MLP

import numpy as np


N, p = 500, 3

evaluate = MLP.random(N=N, p=p)

h = 20


class _Chromosome(BinaryChromosome):

    def decode(self):
        # 0101... --> 0.123
        return IntervalConverter(-5,5)(self)

class ExampleIndividual(MixedIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = FloatChromosome, FloatChromosome, FloatChromosome, _Chromosome

    def decode(self):
        return self.chromosomes[0].reshape(p, h), self.chromosomes[1], self.chromosomes[2], self.chromosomes[3].decode()

    def _fitness(self):
        return evaluate(self.decode())

class ExampleIndividual2(ExampleIndividual, TraitThresholdIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = FloatChromosome, FloatChromosome, FloatChromosome, _Chromosome, FloatChromosome

if __name__ == '__main__':

    MyPopulation = StandardPopulation[ExampleIndividual2]
    pop = MyPopulation.random(n_individuals=100, sizes=[h*p, h, h, 8, 4])
    pop1 = pop.clone(type_=HOFPopulation)
    pop2 = pop.clone()

    pop.mate_prob = pop.mutate_prob = 1

    data = pop.evolve(n_iter=300, history=True, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness', 
        'mean threshold': lambda pop: np.mean([ind.threshold for ind in pop.individuals]),
        'mean mate_prob': lambda pop: np.mean([ind.mate_prob for ind in pop.individuals]),
        'mean mutate_prob': lambda pop: np.mean([ind.mutate_prob for ind in pop.individuals]),
        'best threshold': lambda pop: pop.best_individual.threshold,
        'best mate_prob': lambda pop: pop.best_individual.mate_prob,
        'worst mate_prob': lambda pop: pop.worst_individual.mate_prob,
        'best mutate_prob': lambda pop: pop.best_individual.mutate_prob,
        'worst mutate_prob': lambda pop: pop.worst_individual.mutate_prob,
        'worst desire': lambda pop: pop.worst_individual.desire,
        'best desire': lambda pop: pop.best_individual.desire
        })
    data.to_csv('h.csv')

    for ind2, ind in zip(pop2.sorted_individuals, pop.sorted_individuals):
        ind2.mate_prob = ind.mate_prob
        ind2.mutate_prob = ind.mutate_prob

    data2 = pop2.evolve(n_iter=300, history=True, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'})


    pop1.mate_prob = 0.8
    pop1.mutate_prob = 0.4

    data1 = pop1.evolve(n_iter=300, history=True, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'})
    # pop2.mate_prob = data.loc[300, 'best mate_prob']
    # pop2.mutate_prob = data.loc[300, 'best mutate_prob']
    # data2 = pop2.evolve(n_iter=300, history=True, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'})

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(121)
    data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax, style='--')
    data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax, style='-.')
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)

    ax.legend(('Traditional','Traditional best', 'Traditional2','Traditional2 best', 'New', 'New best'))
    ax.set_xlabel('Generation')

    ax2 = fig.add_subplot(122)
    data[['best mate_prob', 'best mutate_prob', 'best desire', 'worst mate_prob', 'worst mutate_prob', 'worst desire']].plot(ax=ax2)
    plt.show()
    plt.show()

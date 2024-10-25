#!/usr/bin/env python3

"""Variants of Population classes

A population is defined as a container of individuals.

StandardPopulation: Standard Genetic Algorithm
HOFPopulation: Standard Genetic Algorithm with hall of fame
"""

from operator import methodcaller, attrgetter
from random import gauss, random

import numpy as np

from .base import BasePopulation, BaseIndividual
from .individual import binaryIndividual
from .chromosome import BinaryChromosome
from .meta import MetaList

from .deco import side_effect


class StandardPopulation(BasePopulation):
    """Standard GA
    
    Extends:
        BasePopulation

    Params:
        n_elders: the number (or rate) of the last generation
    """

    params = {'n_elders': 0.5}
    
    def transition(self, *args, **kwargs):
        """
        Transitation in Standard GA
        """
        elder = self.get_best_individuals(self.n_elders, copy=True)
        super().transition(*args, **kwargs)
        self.extend(elder)


Population = StandardPopulation


class HOFPopulation(BasePopulation):
    """GA with hall of fame
    
    Extends:
        StandardPopulation
    
    Attributes:
        hall_of_fame (list): the copy of the best individuals
    """

    params = {'hof_size': 2}
    alias ={'hof': 'hall_of_fame'}
    
    # hall_of_fame = []

    def init(self):
        super().init()
        self.hall_of_fame = self.get_best_individuals(self.hof_size, copy=True)

    def transition(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """

        self.mate_with(self.hall_of_fame)
        super().transition(*args, **kwargs)
        self.update_hall_of_fame()

    def update_hall_of_fame(self):
        """Update the hall of fame

        insert the best individuals of the population into the hall of fame;
        meanwhile, remove the worst ones in the hall of fame
        """
        for ind in self:
            for k in range(self.hof_size):
                if self.hall_of_fame[-k-1].fitness < ind.fitness:
                    self.hall_of_fame.insert(self.hof_size-k, ind.copy())
                    self.hall_of_fame.pop(0)
                    break

    @property
    def max_fitness(self):
        if self.hall_of_fame:
            return max(map(attrgetter('fitness'), self.hall_of_fame))
        else:
            return super().max_fitness

    @property
    def best_individual(self):
        if self.hall_of_fame:
            return self.hall_of_fame[np.argmax([_.fitness for _ in self.hall_of_fame])]
        else:
            return super().best_individual


class DualPopulation(StandardPopulation):
    """Dual Genetic Algo.
    
    Extends:
        StandardPopulation
    """

    params ={
    'dual_prob': 0.2,
    'n_elders': 0.3}

    def dual(self):
        # make dual population
        for k, ind in enumerate(self):
            if random() < self.dual_prob:
                d = ind.dual()
                if d.fitness > ind.fitness:
                    self[k] = d
    
    def transition(self, *args, **kwargs):
        self.dual()
        super().transition(*args, **kwargs)


class GamogenesisPopulation(HOFPopulation):
    """Gamogenesis Genetic Algo.
    
    Extends:
        HOFPopulation
    """

    def mate(self, mate_prob=None):
        """Mate the male sub-population and female sub-population.

        Keyword Arguments:
            mate_prob {number} -- the proba. of mating of two individuals (default: {None})
        """

        mate_prob = mate_prob or self.mate_prob
        offspring = [individual.cross(other) for individual, other in zip(self.males, self.females)
        if random() < mate_prob]
        self.individuals.extend(offspring)
        return offspring

    def get_homosex(self, x=0):
        # get the list of individuals with gender `x`
        return [i for i in self if i.gender==x]


class EliminationPopulation(BasePopulation):

    def transition(self, *args, **kwargs):
        elder = self.clone()
        elder.select()
        super().transition(*args, **kwargs)
        self.eliminate()
        self.merge(elder)

    def eliminate(self):
        # remove some individuals randomly from the population
        for individual in self:
            if random() < individual.eliminate_prob():
                self.remove(individual)


class AgePopulation(EliminationPopulation):

    def transition(self, *args, **kwargs):
        for individual in self:
            individual.age += 1
        super().transition(*args, **kwargs)

    def eliminate(self):
        # remove some old individuals
        for individual in self:
            if random() * individual.life_span < individual.age:
                self.remove(individual)


class LocalSearchPopulation(StandardPopulation):
    """Population with `local_search` method
    """

    params = {'n_local_iter': 2}

    # def init(self):
    #     for i in self:
    #         i.max_iter = self.n_local_iter

    def transition(self, *args, **kwargs):
        """Transitation of the states of population

        Calling `local_search` method
        """

        super().transition(*args, **kwargs)
        self.local_search(max_iter=self.n_local_iter)


class ModifiedPopulation(StandardPopulation):

    params = {'mutate_prob_ub':0.5, 'mutate_prob_lb':0.1}

    def mutate(self):
        """Mutate the whole population.
        """
        fm = self.max_fitness
        fa = self.mean_fitness
        for individual in self:
            f = individual.fitness
            if f > fa:
                mutate_prob = self.mutate_prob_ub - (self.mutate_prob_ub-self.mutate_prob_lb) * (f-fa) / (fm-fa)
            else:
                mutate_prob = self.mutate_prob_ub
            if random() < mutate_prob:
                individual.mutate()


def makeBinaryPopulation(n_individuals=20, size=8, as_chromosome=True, cls=None):
    """Make a binary population
    
    Args:
        n_individuals (int, optional): the number of the individuals in the population
        size (int, optional): the size of an individual or a chromosome
        as_chromosome (bool, optional): take chromosomes as the individuals of the population
        cls (None, optional): the type of the population (HOFPopulation by default)
    
    Returns:
        subclass of BasePopulation
    """

    cls = cls or HOFPopulation
    if as_chromosome:
        return cls[BinaryChromosome // size] // n_individuals
    else:
        return cls[binaryIndividual(size)] // n_individuals

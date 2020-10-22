#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulation, random
from .utils import gauss, random
from . import MetaList

class SGAPopulation(BasePopulation):
    """Standard Genetic Algo I.
    
    Extends:
        BasePopulation
    """

    params = {'n_elders': 0.5}
    
    def transit(self, k=None, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """

        elder = self.__class__(self.get_best_individuals(self.n_elders * self.default_size)).clone()
        super(SGAPopulation, self).transit(*args, **kwargs)
        self.merge(elder)


class DualPopulation(BasePopulation):
    """Dual Genetic Algo.
    
    Extends:
        BasePopulation
    """

    params ={'dual_prob': 0.2, 'n_elders': 0.3}

    def dual(self):
        for k, ind in enumerate(self.individuals):
            if random() < self.dual_prob:
                d = ind.dual()
                if d.fitness > ind.fitness:
                    self.individuals[k] = d
    
    def transit(self, k=None, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """
        self.dual()
        elder = self.clone()
        elder.get_best_individuals(self.n_elders)
        super(DualPopulation, self).transit(*args, **kwargs)
        self.merge(elder, select=True)


class GamogenesisPopulation(SGAPopulation):
    # element_class = GenderPopulation

    """Gamogenesis Genetic Algo.
    
    Extends:
        BasePopulation
    """

    def mate(self, mate_prob=None):
        """Mate the whole population.

        Just call the method `mate` of each individual (customizing anthor individual)
        
        Keyword Arguments:
            mate_prob {number} -- the proba. of mating of two individuals (default: {None})
        """

        offspring = [individual.cross(other) for individual, other in zip(self.males, self.females)
        if random() < (mate_prob or self.mate_prob)]
        self.individuals.extend(offspring)
        self.offspring = self.__class__(offspring)

    def get_homosex(self, x=0):
        return [i for i in self.individuals if i.gender==x]


class EliminationPopulation(BasePopulation):
    def transit(self, k=None, *args, **kwargs):
        elder = self.clone()
        elder.select(k)
        super(EliminationPopulation, self).transit(*args, **kwargs)
        self.eliminate()
        self.merge(elder, select=True)

    def eliminate(self):
        for individual in self.individuals:
            if random() < individual.eliminate_prob():
                self.remove(individual)


class AgePopulation(EliminationPopulation):
    def eliminate(self):
        for individual in self.individuals:
            individual.age += 1
            x = individual.age / individual.life_span
            if random() < x:
                self.remove(individual)


class LocalSearchPopulation(BasePopulation):
    '''LocalSearchPopulation'''

    def transit(self, mutate_prob=0.3, mate_prob=0.7):
        """
        Transitation of the states of population by SGA
        """
        c = self.clone()
        c.select(k=0.7)
        self.select()
        if random() < mate_prob:
            self.mate()
        if random() < mutate_prob:
            self.mutate()
        self.merge(c, select=True)
        self.local_search()

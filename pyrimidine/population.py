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


class SGA2Population(SGAPopulation):
    """Standard Genetic Algo II.

    With hall of fame
    
    Extends:
        BasePopulation
    """

    params = {'fame_size':4}

    def init(self):
        self.halloffame = self.get_best_individuals(self.fame_size)

    def transit(self, k=None, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """
        super(SGA2Population, self).transit(*args, **kwargs)
        self.update_halloffame()
        self.add_individuals([i.clone() for i in self.halloffame])

    def update_halloffame(self):
        b = self.best_individual
        for i in self.halloffame:
            if i.fitness < b.fitness:
                self.halloffame.remove(i)
                self.halloffame.append(b.clone())
                break

    @property
    def best_fitness(self):
        return max(_.fitness for _ in self.halloffame)


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
        self.merge(elder)


class GamogenesisPopulation(SGA2Population):
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
        mate_prob = mate_prob or self.mate_prob
        offspring = [individual.cross(other) for individual, other in zip(self.males, self.females)
        if random() < mate_prob]
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
        self.merge(elder)

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
    '''LocalSearchPopulation Class
    
    Population with `local_search` method
    '''

    def transit(self, mutate_prob=0.3, mate_prob=0.7):
        """Transitation of the states of population

        Calling `local_search` method
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulation, random
from .utils import gauss, random
from . import MetaList


class StandardPopulation(BasePopulation):
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
        super(StandardPopulation, self).transit(*args, **kwargs)
        self.merge(elder)


class SGAPopulation(StandardPopulation):
    print('Use StandardPopulation in future!')
    pass


class HOFPopulation(StandardPopulation):
    """Standard Genetic Algo II.

    With hall of fame
    
    Extends:
        StandardPopulation
    """

    params = {'fame_size': 2}

    def init(self):
        self.hall_of_fame = self.get_best_individuals(self.fame_size)

    def postprocess(self, k=None, *args, **kwargs):
        """
        Update the hall of fame after one step of evolution
        """
        self.update_hall_of_fame()
        self.add_individuals([i.clone() for i in self.hall_of_fame])

    def update_hall_of_fame(self):
        b = self.best_individual
        for k, i in enumerate(self.hall_of_fame[::-1]):
            if i.fitness < b.fitness:
                self.hall_of_fame.pop(-k-1)
                self.hall_of_fame.insert(-k-1, b.clone())
                break

    # def update_hall_of_fame(self, n=2):
    #     bs = self.get_best_individuals(n)
    #     N = len(self.hall_of_fame)
    #     k = N
    #     for b in bs[::-1]:
    #         while k > 0:
    #             k -= 1
    #             i = self.hall_of_fame[k]
    #             if i.fitness < b.fitness:
    #                 self.hall_of_fame.pop(k)
    #                 self.hall_of_fame.insert(k, b.clone())
    #                 break

    @property
    def best_fitness(self):
        return max(_.fitness for _ in self.hall_of_fame)


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


class GamogenesisPopulation(HOFPopulation):
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
        self.mate()
        self.mutate()
        self.merge(c, select=True)
        self.local_search()


class ModifiedPopulation(BasePopulation):
    params = {'mutate_prob_ub':0.5, 'mutate_prob_lb':0.1}
    def mutate(self):
        fm = self.best_individual.fitness
        fa = self.mean_fitness
        for individual in self.individuals:
            f = individual.fitness
            if f > fa:
                mutate_prob = self.mutate_prob_ub - (self.mutate_prob_ub-self.mutate_prob_lb) * (f-fa) / (fm-fa)
            else:
                mutate_prob = self.mutate_prob_ub
            if random() < mutate_prob:
                individual.mutate()


# class SelfAdaptivePopulation(SGA2Population):
#     """docstring for SelfAdaptivePopulation"""
#     element_class = SelfAdaptiveIndividual

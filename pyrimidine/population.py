#!/usr/bin/env python3

"""Variants of Population classes

StandardPopulation: Standard Genetic Algorithm
HOFPopulation: Standard Genetic Algorithm with hall of fame
"""

from operator import methodcaller, attrgetter
import numpy as np

from .base import BasePopulation
from .utils import gauss, random
from .meta import MetaList


class StandardPopulation(BasePopulation):
    """Standard Genetic Algo I.
    
    Extends:
        BasePopulation
    """

    params = {'n_elders': 0.5}
    
    def transition(self, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """

        elder = self.get_best_individuals(self.n_elders * self.default_size, copy=True)
        super().transition(*args, **kwargs)
        self.merge(elder, n_sel=self.default_size)

Population = StandardPopulation


class HOFPopulation(StandardPopulation):
    """Standard Genetic Algo With hall of fame
    
    Extends:
        StandardPopulation
    
    Attributes:
        hall_of_fame (list): the best individuals
    """

    params = {'hof_size': 2}
    hall_of_fame = []

    def init(self):
        self.hall_of_fame = self.get_best_individuals(self.hof_size)

    def transition(self, *args, **kwargs):
        """
        Update the hall of fame after each step of evolution
        """

        super().transition(*args, **kwargs)
        self.update_hall_of_fame()
        self.add_individuals(map(methodcaller('clone'), self.hall_of_fame))

    def update_hall_of_fame(self):
        hof_size = len(self.hall_of_fame)
        for ind in self:
            for k in range(hof_size):
                if self.hall_of_fame[-k-1].fitness < ind.fitness:
                    self.hall_of_fame.insert(hof_size-k, ind.clone())
                    self.hall_of_fame.pop(0)
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
        if self.hall_of_fame:
            return max(map(attrgetter('fitness'), self.hall_of_fame))
        else:
            return super().best_fitness

    @property
    def best_individual(self):
        if self.hall_of_fame:
            return self.hall_of_fame[np.argmax([_.fitness for _ in self.hall_of_fame])]
        else:
            return super().best_individual


class DualPopulation(StandardPopulation):
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
    
    def transition(self, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """
        self.dual()
        elder = self.clone()
        elder.get_best_individuals(self.n_elders)
        super().transition(*args, **kwargs)
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
    def transition(self, k=None, *args, **kwargs):
        elder = self.clone()
        elder.select(k)
        super().transition(*args, **kwargs)
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


class LocalSearchPopulation(StandardPopulation):
    '''LocalSearchPopulation Class
    
    Population with `local_search` method
    '''

    def transition(self, *args, **kwargs):
        """Transitation of the states of population

        Calling `local_search` method
        """
        super().transition(*args, **kwargs)
        self.local_search()


class ModifiedPopulation(StandardPopulation):
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


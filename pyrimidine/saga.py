#!/usr/bin/env python3

"""
Self Adaptive GA
"""

from . import *
from .utils import *


class BaseSelfAdaptiveIndividual(MixedIndividual):

    @property
    def mutate_prob(self):
        raise NotImplementedError

    @mutate_prob.setter
    def mutate_prob(self, v):
        raise NotImplementedError

    @property
    def mate_prob(self):
        raise NotImplementedError

    @mate_prob.setter
    def mate_prob(self, v):
        raise NotImplementedError

    @property
    def desire(self):
        raise NotImplementedError


class SelfAdaptiveIndividual(BaseSelfAdaptiveIndividual):
    """Individual for Self-adaptive GA

    Provide at least 2 chromosomes, one of them is coded by float numbers
    representing the prob of mutation and mating.

    Extends:
        BaseSelfAdaptiveIndividual
    """
    
    # By default, the second chromosome (a float vector) encodes the prob of mutation and mating
    element_class = BinaryChromosome, FloatChromosome

    def mutate(self):
        if random() < self.mutate_prob:
            super().mutate()

    def mate(self, other, mate_prob=None):
        if random() < (mate_prob or (self.mate_prob + other.mate_prob)/2):
            return super().mate(other)
        else:
            return self

    @property
    def mutate_prob(self):
        return self.chromosomes[-1][0]

    @mutate_prob.setter
    def mutate_prob(self, v):
        self.chromosomes[-1][0] = v

    @property
    def mate_prob(self):
        return self.chromosomes[-1][1]

    @mate_prob.setter
    def mate_prob(self, v):
        self.chromosomes[-1][1] = v

    @property
    def desire(self):
        return self.chromosomes[-1][2]
    

class RankingIndividual(SelfAdaptiveIndividual):
    ranking = None

    # def mate(self, other, mate_prob=None):
    #     if other.ranking:
    #         if self.threshold <= other.ranking:
    #             return super().mate(other, mate_prob=1)
    #         elif self.threshold <= 2* other.ranking:
    #             return super().mate(other, mate_prob=0.8)
    #         else:
    #             return super().mate(other, mate_prob=0.5)
    #     else:
    #         return super().mate(other)

    # @property
    # def threshold(self):
    #     return self.chromosomes[-1][-1]

def lim(r, e):
    e -= 0.0001
    return 0 if r <= e else (r - e)**2 / (1 - e)

class SSAPopulation(HOFPopulation):

    def transit(self, *args, **kwargs):
        """Transitation of the states of population

        It is considered to be the standard flow of the Genetic Algorithm
        """
        self.mate()
        self.mutate()
        self.doom()
        
        for individual in self.individuals:
            individual.age += 1
        if self.is_crowd():
            self.select(0.3)

    def doom(self):
        # Remove the dead individuals
        self.individuals = [individual for individual in self.individuals if not individual.is_dead()]

    def mate(self):
        self.rank()
        children = []
        for individual, other in zip(self.individuals[:-1], self.individuals[1:]):
            if random() < min(individual.cross_prob, other.cross_prob):
                if self.match(individual, other):
                    children.append(individual.cross(other))
        self.add_individuals(children)

    @classmethod
    def match(cls, individual, other):
        if individual.label != other.label:
            a, b, c = lim(other.ranking, individual.expect), lim(individual.ranking, other.expect), abs(individual.age - other.age)
            p = 1 - c/(20*min(a, b)+1)
            return random() < p
        else:
            return random() < 0.05

    def is_crowd(self):
        return len(self) > 8 * self.default_size

    def select(self, p=0.5):
        # self.select(n_sel=p)
        self.individuals = choice_with_fitness(self.individuals, fs=None, n=int(self.n_individuals*p))
        # self.individuals = [individual for individual in self.individuals if random() < p]

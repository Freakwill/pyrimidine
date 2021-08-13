#!/usr/bin/env python3

from .base import BaseIndividual, BaseChromosome
from .population import  HOFPopulation
from .chromosome import FloatChromosome
from .utils import random, choice_with_fitness

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

    def doom(self):
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


    def postprocess(self):
        for individual in self.individuals:
            individual.age += 1
        if self.is_crowd():
            self.select(0.3)
        super(SSAPopulation, self).postprocess()

    def is_crowd(self):
        return len(self) > 8 * self.default_size

    def select(self, p=0.5):
        # self.select(n_sel=p)
        self.individuals = choice_with_fitness(self.individuals, fs=None, n=int(self.n_individuals*p))
        # self.individuals = [individual for individual in self.individuals if random() < p]


class SSAIndividual(BaseIndividual):
    pass


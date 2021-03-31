#!/usr/bin/env python3

from .base import BaseIndividual, BaseChromosome
from .population import  HOFPopulation
from .chromosome import FloatChromosome
from .utils import random

def lim(r, e):
    e -= 0.0001
    return 0 if r < e else (r - e)**2 / (1 - e)

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
        for individual in self.individuals:
            for other in self.individuals:
                if any(individual.chromosomes[0] != other.chromosomes[0]) and random() < min(individual.cross_prob, other.cross_prob):
                    if self.match(individual, other):
                        children.append(individual.cross(other))
        self.add_individuals(children)

    def match(self, individual, other):
        return random() < min(lim(other.ranking, individual.expect), lim(individual.ranking, other.expect))

    def postprocess(self):
        for individual in self.individuals:
            individual.age += 1
        if self.is_crowd():
            self.drop(0.5)
        super(SSAPopulation, self).postprocess()

    def is_crowd(self):
        return len(self) > 3 * self.default_size


class SSAIndividual(BaseIndividual):
    pass


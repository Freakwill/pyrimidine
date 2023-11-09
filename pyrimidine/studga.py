#!/usr/bin/env python

"""The Stud GA

References:
Khatib, Wael and Peter John Fleming. “The Stud GA: A Mini Revolution?” Parallel Problem Solving from Nature (1998).
"""

from .population import HOFPopulation
from .utils import random, choice

class StudPopulation(HOFPopulation):

    params = {'fame_size': 2}
    
    def mate(self, mate_prob=None):
        """Mating in studGA

        individuals only mate with individuals in the hall of fame
        """

        mate_prob = mate_prob or self.mate_prob
        offspring = []
        for individual in self.individuals:
            if individual in self.halloffame:
                continue
            if random() < (mate_prob or self.mate_prob):
                other = choice(self.halloffame)
                offspring.append(individual.cross(other))
        self.individuals.extend(offspring)
        self.offspring = self.__class__(offspring)

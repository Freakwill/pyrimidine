#!/usr/bin/env python

"""The Stud GA

References:
Khatib, Wael and Peter John Fleming. “The Stud GA: A Mini Revolution?” Parallel Problem Solving from Nature (1998).
"""

from random import random

from .population import HOFPopulation
from .utils import choice_uniform


class StudPopulation(HOFPopulation):

    params = {'fame_size': 2}
    
    def mate(self, mate_prob=None):
        """Mating in studGA

        individuals only mate with individuals in the hall of fame
        """

        mate_prob = mate_prob or self.mate_prob
        offspring = []
        for individual in self:
            if individual in self.halloffame:
                continue
            if random() < (mate_prob or self.mate_prob):
                other = choice_uniform(self.halloffame)
                offspring.append(individual.cross(other))
        self.extend(offspring)
        return offspring

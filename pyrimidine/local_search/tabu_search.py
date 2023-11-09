#!/usr/bin/env python3

"""
Tabu Search was created by Fred W. Glover in 1986 and formalized in 1989
"""

from ..utils import random, choice
from pyrimidine.individual import MemoryIndividual


class BaseTabuSearch(MemoryIndividual):
    """Tabu Search algorithm
    """

    params = {'value': 0,
        'tabu_list': [],
        'actions': [],
        'tabu_size': 10
        }

    def init(self):
        self.memory = self.clone()
        self.best_fitness = self.memory.fitness

    def transit(self, *args, **kwargs):
        action = choice(self.actions)
        cpy = self.get_neighbour(action)
        if action not in self.tabu_list:
            if cpy.fitness > self.best_fitness:
                self.chromosomes = cpy.chromosomes
                self.best_fitness = cpy.fitness
            else:
                if random() < 0.02:
                    self.backup()
                    self.chromosomes = cpy.chromosomes
                else:
                    self.tabu_list.append(action)
        else:
            if cpy.fitness > self.best_fitness:
                self.chromosomes = cpy.chromosomes
                self.best_fitness = cpy.fitness
                self.tabu_list.remove(action)
        self.update_tabu_list()

    def update_tabu_list(self):
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def get_neighbour(self, action):
        raise NotImplementedError


class SimpleTabuSearch(BaseTabuSearch):
    def get_neighbour(self, action):
        cpy = self.clone()
        i, j = action
        cpy.chromosomes[i][j] = cpy.gene.random()
        return cpy

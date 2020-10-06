#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from random import random, choice
from pyrimidine.base import BaseFitnessModel


class BaseTabuSearch(BaseFitnessModel):
    """Tabu Search algorithm
    """

    params = {'value': 0,
        'tabu_list': [],
        'actions': None}

    def init(self):
        self.best_fitness = self.fitness

    def transit(self, *args, **kwargs):
        action = choice(self.__class__.actions)
        cpy = self.move(action)
        if action not in self.tabu_list:
            if cpy.fitness < self.best_fitness:
                if random() < 0.01:
                    self.chromosomes = cpy.chromosomes
                    self.tabu_list.append(action)
            else:
                if cpy.fitness > self.best_fitness:
                    self.best_fitness = cpy.fitness
                self.chromosomes = cpy.chromosomes
            return
        elif cpy.fitness > self.best_fitness:
            self.chromosomes = cpy.chromosomes
            self.tabu_list.remove(action)
            self.best_fitness = cpy.fitness
            return

    def move(self, action):
        raise NotImplementedError



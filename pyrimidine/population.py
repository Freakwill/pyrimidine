#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulation, random
from .utils import gauss

class SGAPopulation(BasePopulation):
    """Standard Genetic Algo II.
    
    Extends:
        BasePopulation
    """
    
    def transitate(self, k=None, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """
        elder = self.clone()
        elder.select(k)
        super(SGAPopulation, self).transitate(*args, **kwargs)
        self.merge(elder, select=True)


class EliminationPopulation(BasePopulation):
    def transitate(self, k=None, *args, **kwargs):
        elder = self.clone()
        elder.select(k)
        super(EliminationPopulation, self).transitate(*args, **kwargs)
        self.eliminate()
        self.merge(elder, select=True)

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
    '''[Summary for Class LocalSearchPopulation]'''
    def transitate(self, mutate_prob=0.3, mate_prob=0.7):
        """
        Transitation of the states of population by SGA
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

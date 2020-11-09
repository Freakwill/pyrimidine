#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import BaseSpecies
from .utils import  *

from itertools import product

class DualSpecies(BaseSpecies):
    params = {'n_elders':0.5, 'mate_prob':0.75}

    @property  
    def male_population(self):
        return self.populations[0]

    @property  
    def female_population(self):
        return self.populations[1]

    @property  
    def males(self):
        return self.populations[0].individuals

    @property  
    def females(self):
        return self.populations[1].individuals

    @property
    def male_fitness(self):
        return self.populations[0].fitness

    @property
    def female_fitness(self):
        return self.populations[1].fitness

    def mate(self):
        self.populations[0].rank(tied=True)
        self.populations[1].rank(tied=True)
        children = [male.cross(female) for male, female in product(self.males, self.females) if random() < self.mate_prob and self.match(male, female)]
        self.populations[0].add_individuals(children[::2])
        self.populations[1].add_individuals(children[1::2])


    def match(self, male, female):
        return True


    def transit(self, *args, **kwargs):
        elder = self.__class__([
            self.populations[0].__class__(self.populations[0].get_best_individuals(self.n_elders * self.populations[0].default_size)),
            self.populations[1].__class__(self.populations[1].get_best_individuals(self.n_elders * self.populations[1].default_size))
            ]).clone()
        self.select()
        self.mate()
        self.mutate()
        self.merge(elder)

    def select(self):
        self.populations[0].select()
        self.populations[1].select()

    def mutate(self):
        self.populations[0].mutate()
        self.populations[1].mutate()

    def merge(self, other):
        self.populations[0].merge(other.populations[0])
        self.populations[1].merge(other.populations[1])


    def post_process(self):
        super(DualSpecies, self).post_process()
        self.populations[0].fitness = self.populations[1].fitness = None


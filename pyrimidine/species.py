#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import BaseSpecies
from .utils import  *

class DualSpecies(BaseSpecies):

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
        self.populations[0].rank()
        self.populations[1].rank()
        male_offspring = []
        female_offspring = []
        for _ in range(2):
            shuffle(self.females)
            for male, female in zip(self.males, self.females):
                if self.match(male, female):
                    child = male.cross(female)
                    if random()<0.5:
                        male_offspring.append(child)
                    else:
                        female_offspring.append(child)

        self.populations[0].individuals += male_offspring
        self.populations[1].individuals += female_offspring


    def match(self, male, female):
        return True


    def transit(self, *args, **kwargs):
        self.populations[0].select()
        self.populations[1].select()
        self.mate()
        self.populations[0].mutate()
        self.populations[1].mutate()


    def post_process(self):
        super(DualSpecies, self).post_process()
        self.populations[0].fitness = self.populations[1].fitness = None


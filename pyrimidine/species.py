#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import BaseSpecies

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
        offspring = [male.cross(female) for male, female in zip(self.males, self.females) if self.match(male, female)]
        self.populations[0].individuals += offspring
        offspring = [male.cross(female) for male, female in zip(self.males, self.females) if self.match(male, female)]
        self.populations[1].individuals += offspring
        

    def match(self, male, female):
        return True


    def transitate(self, *args, **kwargs):
        self.populations[0].select()
        self.populations[1].select()
        self.mate()
        self.populations[0].mutate()
        self.populations[1].mutate()
        self.populations[0].ranking()
        self.populations[1].ranking()
        self.populations[0].fitness = self.populations[1].fitness = None

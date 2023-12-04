#!/usr/bin/env python3


from itertools import product
import threading

from . import BaseMultiPopulation
from .utils import *


class MultiPopulation(BaseMultiPopulation):
    pass


class DualPopulation(BaseMultiPopulation):

    params = {'n_elders':0.5, 'mate_prob':0.75}

    default_size = 2

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
        children = []
        def _target(male, female):
            if random()<0.5:
                child = male.cross(female)
            else:
                child = female.cross(male)
            children.append(child)
        ps = [threading.Thread(target=_target, args=(male, female)) for male, female in product(self.males, self.females) if random() < self.mate_prob and self.match(male, female)]

        for p in ps:
            p.start()
        for p in ps:
            p.join()

        self.populations[0].extend(children[::2])
        self.populations[1].extend(children[1::2])

    def match(self, male, female):
        return True

    def transition(self, *args, **kwargs):
        elder = self.__class__([
            self.populations[0].get_best_individuals(self.n_elders * self.populations[0].default_size, copy=True),
            self.populations[1].get_best_individuals(self.n_elders * self.populations[1].default_size, copy=True)
            ])
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


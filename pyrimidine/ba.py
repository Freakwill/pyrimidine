#!/usr/bin/env python

"""
Bat Algorithm
"""

from . import *
from .utils import *

class Bat(MemoryIndividual):
    element_class = FloatChromosome
    default_size = 5
    memory ={'position':None,
    'fitness':None}

    params = {'frequency': 0.5,
        'pulse_rate': 0,
        'loudness': 0
        }


    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.fitness = None

    @property
    def velocity(self):
        return self.chromosomes[-1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[-1] = v

    @property
    def best_position(self):
        return self.chromosomes[0]

    @best_position.setter
    def best_position(self, x):
        self.chromosomes[0] = x
        self.fitness = None

    def move(self):
        self.position += self.velocity


class Bats(HOFPopulation):

    alias = {'bats': 'elements'}

    params = {
        'gamma': 0.5,
        'alpha': 0.95,
        'pulse_rate': 0.9
        }


    def transit(self, k):
        
        for bat in self:
            bat.velocity += (self.best_individual.position - bat.position) * bat.frequency
            bat.move()

            # local search
            for i in range(len(bat.position)):
                if random() < self.pulse_rate:
                    bat.position[i] += bat.loudness * uniform(-1, 1)

            # update the params
            bat.frequency = uniform(0, .2)
            bat.pulse_rate = self.pulse_rate * (1 - exp(-self.gamma * k))
            bat.loudness *= self.alpha
            if random() < bat.loudness:
                bat.backup()
        self.update_hall_of_fame()


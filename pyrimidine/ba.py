#!/usr/bin/env python

"""
The Bat Algorithm is a nature-inspired optimization algorithm developed by Xin-She Yang in 2010.
It is based on the echolocation behavior of bats.
Bats emit ultrasonic pulses and listen to the echoes to determine the distance to obstacles and the location of prey.
This behavior forms the basis of the algorithm where solutions are represented as virtual bats,
and their positions in the search space are adjusted iteratively to find the optimal solution. 
"""

from . import *
from .utils import *

class Bat(MemoryIndividual):
    element_class = FloatChromosome
    default_size = 5

    _memory = {'position':None,
        'fitness':None
        }

    params = {'frequency': 0.5,
        'pulse_rate': 0,
        'loudness': 0,
        'scale': 0.2
        }


    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.__fitness = None

    @property
    def velocity(self):
        return self.chromosomes[-1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[-1] = v

    def move(self):
        self.position += self.velocity * self.scale


class Bats(HOFPopulation):

    alias = {'bats': 'elements'}

    params = {
        'gamma': 0.5,
        'alpha': 0.95,
        'pulse_rate': 0.9
        }
 
    def init(self):
        for bat in self:
            bat.backup(check=False)
        super().init()

    def transition(self, k):
        
        for bat in self:
            bat.frequency = uniform(0, .1)
            bat.velocity += (self.best_individual.memory['position'] - bat.position) * bat.frequency
            bat.move()

            # local search
            for i, pi in enumerate(bat.position):
                if random() < bat.pulse_rate:
                    bat.position[i] = pi + bat.loudness * uniform(-1, 1)

            # update the params
            bat.pulse_rate = self.pulse_rate * (1 - exp(-self.gamma * k))
            bat.loudness *= self.alpha
            if random() > bat.pulse_rate:
                bat.backup()
            if random() < 0.1:
                self.update_hall_of_fame()
        self.update_hall_of_fame()


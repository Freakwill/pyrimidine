#!/usr/bin/env python3

"""
The Bat Algorithm is a nature-inspired optimization algorithm developed by Xin-She Yang in 2010.
It is based on the echolocation behavior of bats.
Bats emit ultrasonic pulses and listen to the echoes to determine the distance to obstacles and the location of prey.
This behavior forms the basis of the algorithm where solutions are represented as virtual bats,
and their positions in the search space are adjusted iteratively to find the optimal solution.

*References*
Gagnon, Iannick et al. “A critical analysis of the bat algorithm.” Engineering Reports 2 (2020): n. pag.
Yang, Xin-She. “A New Metaheuristic Bat-Inspired Algorithm.” Nature Inspired Cooperative Strategies for Optimization (2010).
Yang, Xin-She and Amir Hossein Gandomi. “Bat algorithm: a novel approach for global engineering optimization.” Engineering Computations 29 (2012): 464-483.
"""


from math import exp
from random import random

import numpy as np

from ..base import BaseIndividual, BasePopulation
from ..chromosome import FloatChromosome
from ..pso import BaseParticle


class Bat(BaseParticle):

    params = {'frequency': 0.5,
        'pulse_rate': 0,
        'loudness': 0,
        'scale': .1
        }

    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.clear_cache()

    @property
    def velocity(self):
        return self.chromosomes[-1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[-1] = v

    def move(self):
        self.position += self.velocity * self.scale


class Bats(BasePopulation):

    alias = {'bats': 'elements', 'n_bats': 'n_elements'}

    params = {
        'gamma': 0.5,
        'alpha': 0.95,
        'pulse_rate': 0.9,
        'scaling': 0.2
        }
 
    def init(self):
        for bat in self:
            bat.backup(check=False)
            bat.frequency = random()
            bat.init_pulse_rate = random()*0.5 + 0.5
        super().init()

    def transition(self, k):
        max_frequency = max(self.get_all('frequency'))
        min_frequency = min(self.get_all('frequency'))
        mean_loudness = np.mean(list(self.get_all('loudness')))
        for bat in self:
            bat.frequency = random() * (max_frequency-min_frequency) + min_frequency
            bat.velocity += (self.best_individual.memory['position'] - bat.position) * bat.frequency
            bat.move()

            # local search
            for i, pi in enumerate(bat.position):
                if random() > bat.pulse_rate:
                    r = random()*2 - 1
                    bat.position[i] = self.best_individual.memory['position'][i] + mean_loudness * r * self.scaling

            # update the params
            bat.pulse_rate = bat.init_pulse_rate * (1 - exp(-self.gamma * (k+1)))
            bat.loudness *= self.alpha
            if random() < bat.loudness:
                bat.backup()


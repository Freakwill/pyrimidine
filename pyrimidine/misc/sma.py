#!/usr/bin/env python3

"""Slime Mould Algorithm
"""

from random import random

from scipy.spatial.distance import pdist, squareform
import numpy as np

from ..base import BaseIndividual
from ..chromosome import FloatChromosome
from ..population import HOFPopulation

from ..deco import basic_memory
from ..utils import randint2


@basic_memory
class SlimyMaterial(BaseIndividual):

    def approach_food(self, fame, direction, p, vb, vc):
        if random() < p:
            self[:] = (fame + vb * direction)[:]
        else:
            self[:] = (vc * self)[:]

    def random_move(self):
        raise NotImplementedError


class SlimeMould(HOFPopulation):
    """Slime Mould Algorithm
    """

    element_class = SlimyMaterial

    params = {
        "max_iter": 100
    }

    def get_ranks(self):
        all_fitness = np.array([i._fitness() for i in self])
        max_fitness = np.max(all_fitness)
        min_fitness = np.min(all_fitness)
        return (max_fitness - all_fitness)/(max_fitness - min_fitness)

    def approach_food(self, t):
        N = len(self)
        # calculate vc and a
        vc = 1 - t/self.max_iter
        a = np.arctanh(vc)
        # all fitness, max/min fitness
        all_fitness = self.get_all_fitness()
        max_fitness = np.max(all_fitness)
        # calculate p and w
        ps = np.tanh(np.subtract(max_fitness, all_fitness))
        ks = np.argsort(all_fitness)
        sign = np.ones(N); sign[ks[:N//2]] = -1
        ws = 1 + sign * random() * np.log1p(self.get_ranks())
        fame = self[ks[-1]]
        for sm, p, w in zip(self, ps, ws):
            i, j = randint2(0, N-1)
            direction = w * self[i] - self[j]
            vb = a * (2*random() - 1)
            if random()< 0.03:
                sm.random_move()
            else:
                sm.approach_food(fame, direction, p, vb, vc)
    
    def transition(self, t):

        self.approach_food(t)
        for i in self:
            i.backup()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from random import random
from pyrimidine.base import BaseIterativeModel
from pyrimidine.utils import metropolis_rule


class SimulatedAnnealing(BaseIterativeModel):
    """Simulated Annealing algorithm
    """

    phantom = None

    params = {'ext_c': 0.995,
        'int_c': 0.996,
        'nepoch': 200,
        'initT': 100,
        'termT': 0.0001}

    def init(self):
        self.phantom = self.clone(fitness=None)

    def transit(self, *args, **kwargs):
        T = self.initT
        for epoch in range(self.nepoch):
            self.phantom.move(T)
            T *= self.int_c
            if T < self.termT:
                break

    def post_process(self):

        self.initT *= self.ext_c
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.fitness = self.phantom.fitness
        

    def move(self, T):
        """Transition of states
        
        Arguments:
            T {number} -- temperature
        """

        cpy = self.get_neighbour()

        # Metropolis rule
        flag = metropolis_rule(D=cpy.fitness - self.fitness, T=T)
        if flag:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness


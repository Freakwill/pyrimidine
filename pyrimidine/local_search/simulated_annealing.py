#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from random import random
from pyrimidine.base import BaseIterativeModel


def metropolis_rule(D, T, epsilon=0.000001):
    
    if D < 0:
        p = math.exp(D/max(T, epsilon))
        if random() < p:
            flag = True
        else:
            flag = False
    else:
        flag = True
    return flag


class SimulatedAnnealing(BaseIterativeModel):
    """Simulated Annealing algorithm
    """

    phantom = None
    params = {'ext_c': 0.99,
        'int_c': 0.995,
        'nepoch': 500,
        'initT': 100,
        'termT': 0.0001}

    def init(self):
        if self.phantom is None:
            self.phantom = self.clone()

    def transit(self, *args, **kwargs):
        T = self.initT
        for epoch in range(self.nepoch):
            self.phantom.move(T)
            T *= self.int_c
            # if T < self.termT:
            #     break

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


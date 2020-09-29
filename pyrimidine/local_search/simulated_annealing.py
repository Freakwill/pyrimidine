#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from random import random
from .base import BaseIterativeModel


class SimulatedAnnealing(BaseIterativeModel):
    """Simulated Annealing algorithm
        
    Arguments:
        state {Individual} -- state of the physical body in annealing
        initT {number} -- initial temperature
    
    Returns:
        state
    """

    params = {'c': 0.995,
        'cc': 0.999,
        'nepoch': 100,
        'initT': 100,
        'ngen': 100}

    def transitate(self, gen):
        T = self.initT
        for epoch in range(self.nepoch):
            self.move(T)
            T *= self.cc ** self.nepoch
        self.initT *= self.c ** gen

    def move(self, T):
        """Transition of states
        
        Arguments:
            T {number} -- temperature
        """

        cpy = self.get_neighbour()

        # Metropolis rule
        D = cpy.fitness - self.fitness
        epsilon = 0.00001
        if D < 0:
            p = min((1, math.exp(D/(T+epsilon))))
            if random() < p:
                self.chromosomes = cpy.chromosomes
        else:
            self.chromosomes = cpy.chromosomes


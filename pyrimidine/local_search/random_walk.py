#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats

from pyrimidine.base import BaseIterativeModel


class RandomWalk(BaseIterativeModel):
    """Random Walk
        
    Arguments:
        state {Individual} -- state of the physical body in annealing
        initT {number} -- initial temperature
    
    Returns:
        state
    """

    params={'mu': 0,
        'sigma': 0.1,
        'gen': 100}

    def transitate(self, gen):
        """Transition of states
        """

        n = scipy.stats.norm(self.mu, self.sigma)
        cpy = self.clone(fitness=None)
        cpy.chromosomes = [chromosome + n.rvs(chromosome.n_genes) for chromosome in cpy.chromosomes]

        # Metropolis rule
        D = cpy.fitness - self.fitness
        if D > 0:
            self.chromosomes = cpy.chromosomes


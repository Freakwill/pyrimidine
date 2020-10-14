#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats

from pyrimidine.base import BaseFitnessModel


class RandomWalk(BaseFitnessModel):
    """Random Walk
        
    Arguments:
        state {Individual} -- state of the physical body in annealing
        initT {number} -- initial temperature
    
    Returns:
        state
    """

    params={'sigma': 1}

    def transit(self, k, *args, **kwargs):
        """Transition of states
        """
        
        sigma *= self.sigma * 0.99**k
        n = scipy.stats.norm(0, sigma)
        cpy = self.clone(fitness=None)
        cpy.chromosomes = [chromosome + n.rvs(chromosome.n_genes) for chromosome in cpy.chromosomes]

        # Metropolis rule
        D = cpy.fitness - self.fitness
        if D > 0:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness


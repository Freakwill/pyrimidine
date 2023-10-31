#!/usr/bin/env python3


import scipy.stats

from .. import FitnessModel


class RandomWalk(FitnessModel):
    """Random Walk
        
    Arguments:
        state {Individual} -- state of the physical body in annealing
        initT {number} -- initial temperature
    
    Returns:
        state
    """

    params = {'sigma': 1}

    def transit(self, k, *args, **kwargs):
        """Transition of states
        """
        
        sigma *= self.sigma * 0.99**k
        n = gauss(0, sigma)
        cpy = self.clone(fitness=None)
        cpy.chromosomes = [chromosome + n.rvs(chromosome.n_genes) for chromosome in cpy.chromosomes]

        D = cpy.fitness - self.fitness
        if D > 0:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness

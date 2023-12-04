#!/usr/bin/env python3


from scipy.stats import norm

from ..mixin import FitnessMixin


class RandomWalk(FitnessMixin):
    """Random Walk Algo.
    """

    params = {'sigma': 1}

    def transit(self, k, *args, **kwargs):
        sigma *= self.sigma * 0.99**k
        cpy = self.copy(fitness=None)
        cpy.mutate(sigma)

        D = cpy.fitness - self.fitness
        if D > 0:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness

    def mutate(self, sigma):
        n = norm(0, sigma)
        self.chromosomes = [chromosome + n.rvs(chromosome.n_genes) for chromosome in cpy]

#!/usr/bin/env python

"""
(mu + lambda) - Evolution Strategy

*References*
Rechenberg, I. 1973. Evolutions strategie – Optimierung technischer Systeme nach Prinzipien der biologischen Evolution, Frommann-Holzboog.
"""


from .base import BasePopulation
from .utils import randint2


class EvolutionStrategy(BasePopulation):
    # Evolution Strategy

    params ={
        "mu" : 10,
        "lambda_": 20
    }

    def init(self):
        super().init()
        if 'mu' not in self.params:
            self.set_params(mu=self.default_size) 

    def transition(self, *args, **kwargs):
        offspring = self.mate()
        self.extend(offspring)
        self.mate()
        self.select_best_individuals(self.mu)

    def mate(self, lambda_=None):
        lambda_ = lambda_ or self.lambda_
        offspring = []
        n = len(self)
        for _ in range(lambda_):
            i, j = randint2(0, n-1)
            child = self[i].cross(self[j])
            offspring.append(child)
        return offspring

    def select_best_individuals(self, mu=None):
        mu = mu or self.mu
        self.individuals = self.get_best_individuals(mu)

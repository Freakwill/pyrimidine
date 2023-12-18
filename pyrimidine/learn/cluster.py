#!/usr/bin/env python3

"""GA for linear regression
"""

import numpy as np
from sklearn.cluster import KMeans

from ..chromosome import FloatChromosome
from ..population import StandardPopulation

from ..learn import BaseEstimator


class GALinearRegression(BaseEstimator, KMeans):
    """Linear Regression by GA
    """

    estimated_params = ('cluster_centers_',)

    @classmethod
    def create_model(cls, *args, **kwargs):
        return KMeans(*args, **kwargs)

    @classmethod
    def config(cls, X, Y=None, n_clusters=2, n_individuals=10, *args, **kwargs):

        n_features = X.shape[1]

        class MyIndividual(FloatChromosome // (n_clusters, n_features)):

            def decode(self):
                model = cls.create_model(*args, **kwargs)
                model.cluster_centers_ = np.asarray(self)
                return model

            def _fitness(self):
                model = self.decode()
                return model.score(X)

        MyPopulation = StandardPopulation[MyIndividual]

        pop = MyPopulation.random(n_individuals=n_individuals)

        return pop

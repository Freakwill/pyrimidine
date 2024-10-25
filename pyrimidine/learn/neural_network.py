#!/usr/bin/env python3

"""GA for optimization of neural networks
"""

import numpy as np

from sklearn.neural_network import MLPRegressor

from .. import MixedIndividual, FloatChromosome, FloatMatrixChromosome
from ..population import StandardPopulation
from ..learn import BaseEstimator


class GAMLPRegressor(BaseEstimator, MLPRegressor):

    """GA for MLP Regression
    """

    hidden_dim = 4
    max_iter = 100
    n_layers = 3

    estimated_params = ('coefs_', 'intercepts_')

    @classmethod
    def create_model(cls, *args, **kwargs):
        # create MLPRegressor object
        model = MLPRegressor(hidden_layer_sizes=(cls.hidden_dim,), max_iter=1, *args, **kwargs)
        model.out_activation_ = 'identity'
        model.n_layers_ = cls.n_layers
        return model

    def __init__(self, *args, **kwargs):
        # initalize MLPRegressor
        super().__init__(hidden_layer_sizes=(self.hidden_dim,), max_iter=1, *args, **kwargs)
        self.out_activation_ = 'identity'
        self.n_layers_ = 3

    @classmethod
    def config(cls, X, Y, n_individuals=10, *args, **kwargs):
        # configure the population for GA based on the data X, Y

        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        class MyIndividual(MixedIndividual):

            element_class = FloatMatrixChromosome, FloatChromosome, FloatMatrixChromosome, FloatChromosome

            def _fitness(self):
                model = self.decode()
                return model.score(X, Y)

            def decode(self):
                model = cls.create_model(*args, **kwargs)
                model.coefs_ = tuple(map(np.asarray, self[::2]))
                model.intercepts_ = tuple(map(np.asarray, self[1::2]))
                model.n_layers_ = 3
                return model

        MyPopulation = StandardPopulation[MyIndividual]

        return MyPopulation.random(n_individuals=n_individuals, size=((input_dim, cls.hidden_dim), cls.hidden_dim, (cls.hidden_dim, output_dim), output_dim))


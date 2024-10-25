#!/usr/bin/env python3

"""GA for linear regression
"""

import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import LinearRegression

from ..chromosome import BinaryChromosome, FloatChromosome
from ..individual import MixedIndividual
from ..population import StandardPopulation

from ..learn import BaseEstimator

from digit_converter import IntervalConverter


c = IntervalConverter(lb=-60, ub=60)
class _BinaryChromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class GALinearRegression(BaseEstimator, LinearRegression):
    
    """Linear Regression by GA
    """

    estimated_params = ('coef_', 'intercept_')

    @classmethod
    def create_model(cls, *args, **kwargs):
        """Create linear regression model
        
        Returns:
            LinearRegression (of scikit-learn)
        """
        return LinearRegression(*args, **kwargs)

    @classmethod
    def config(cls, X, Y, n_individuals=10, *args, **kwargs):
        # configure a population based on the data X and Y

        input_dim = X.shape[1]
        assert np.ndim(Y) == 1, 'only support 1D array for `Y`'
        # output_dim = Y.shape[1]

        class MyIndividual(MixedIndividual):

            element_class = FloatChromosome, _BinaryChromosome

            def decode(self):
                model = cls.create_model(*args, **kwargs)
                model.coef_ = np.asarray(self[0])
                model.intercept_ = self[1].decode()
                return model

            def _fitness(self):
                model = self.decode()
                return model.score(X, Y)

        MyPopulation = StandardPopulation[MyIndividual]

        pop = MyPopulation.random(n_individuals=n_individuals, size=(input_dim, 8))

        return pop

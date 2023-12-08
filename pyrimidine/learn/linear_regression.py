#!/usr/bin/env python3


from pyrimidine import *
from digit_converter import IntervalConverter

c = IntervalConverter(lb=-60, ub=60)
class _BinaryChromosome(BinaryChromosome):
    def decode(self):
        return c(self)


import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import LinearRegression
from ..learn import BaseEstimator


class GALinearRegression(BaseEstimator, LinearRegression):
    '''Linear Regression by GA
    '''

    @classmethod
    def create_model(cls, *args, **kwargs):
        return LinearRegression(*args, **kwargs)

    def _fit(self, X, Y):
        self.pop.ezolve(n_iter=self.max_iter)
        model_ = self.pop.solution
        self.coef_ = model_.coef_
        self.intercept_ = model_.intercept_

    @classmethod
    def config(cls, X, Y, n_individuals=10, *args, **kwargs):

        input_dim = X.shape[1]
        # output_dim = Y.shape[1]

        class MyIndividual(MixedIndividual):
            params={'sigma':0.02}
            element_class = FloatChromosome, _BinaryChromosome

            def decode(self):
                model = cls.create_model(*args, **kwargs)
                model.coef_ = np.asarray(self[0])
                model.intercept_ = self[1].decode()
                return model

            def _fitness(self):
                model = self.decode()
                return model.score(X, Y)

        class MyPopulation(HOFPopulation):
            element_class = MyIndividual

        pop = MyPopulation.random(n_individuals=n_individuals, size=(input_dim, 8))

        return pop

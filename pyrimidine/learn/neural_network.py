#!/usr/bin/env python3

import numpy as np
import numpy.linalg as LA
from scipy.special import softmax
from scipy.stats import entropy

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

from pyrimidine import MixedIndividual, FloatChromosome, FloatMatrixChromosome
from pyrimidine.population import HOFPopulation
from pyrimidine.learn.base import BaseEstimator


class GAANN(BaseEstimator, Sequential):
    """GA for ANN
    """

    pop = None

    max_iter = 100

    @classmethod
    def create_model(cls, input_dim, output_dim):
        # create Sequential object
        hidden_dim = 4
        model = Sequential()
        model.add(Dense(hidden_dim, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim))
        return model

    @classmethod
    def create(cls, input_dim, output_dim):
        # create GAANN object
        hidden_dim = 4
        model = cls()
        model.add(Dense(hidden_dim, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim))
        return model

    @classmethod
    def from_data(cls, X, Y):
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        return cls.create(input_dim, output_dim)

    @classmethod
    def config(cls, X, Y, n_individuals=10):
        hidden_dim = 4
        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        class MyIndividual(MixedIndividual):

            element_class = FloatMatrixChromosome, FloatChromosome, FloatChromosome, FloatChromosome

            def _fitness(self):
                model = self.decode()
                return r2_score(Y, model.predict(X))

            def decode(self):
                model = cls.create_model(input_dim, output_dim)
                for k, layer in enumerate(model.layers):
                    weights = (self.chromosomes[2*k], self.chromosomes[2*k+1])
                    layer.set_weights(weights)
                return model

        MyPopulation = HOFPopulation[MyIndividual]

        return MyPopulation.random(n_individuals=n_individuals, size=((input_dim, hidden_dim), hidden_dim, (hidden_dim, output_dim), output_dim))

    def predict(self, X):
        return super().predict(X, verbose=0)

    def fit(self, X, Y, pop=None):
        self.pop = pop or self.pop or self.config(X, Y)
        self.pop.ezolve(n_iter=self.max_iter)
        model_ = self.pop.solution
        self.set_weights(model_.get_weights())

    def score(self, X, Y):
        return r2_score(Y, self.predict(X))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings, os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from keras.models import Sequential
from keras.layers import Dense

from pyrimidine import MixedIndividual, FloatChromosome, SGAPopulation, FloatMatrixChromosome
from pyrimidine.learn.base import BaseEstimator

import numpy.linalg as LA

class GAANN(BaseEstimator, Sequential):
    """GA for ANN
    """

    @classmethod
    def create_model(cls, X, Y):
        hiden = 4
        model = Sequential()
        model.add(Dense(hiden, activation='relu', input_dim=X.shape[1]))
        model.add(Dense(Y.shape[0]))
        return model

    @classmethod
    def config(cls, X, Y):
        model = cls.create_model(X, Y)
        
        class MyIndividual(MixedIndividual):
            element_class = FloatMatrixChromosome, FloatChromosome, FloatChromosome, FloatChromosome

            def _fitness(self):
                model = self.decode()
                return 1 /(LA.norm(model.predict(X) - Y) +1)

            def decode(self):
                model = GAANN.create_model(X, Y)
                for k, layer in enumerate(model.layers):
                    weights = (self.chromosomes[2*k], self.chromosomes[2*k+1])
                    layer.set_weights(weights)
                return model

        class MyPopulation(SGAPopulation):
            element_class = MyIndividual

        pop = MyPopulation.random(n_individuals=40, sizes=((2,4),4,(4,2),2))
        return pop

    def postprocess(self):
        model = self.best.decode()
        self.set_weights(model.get_weights())


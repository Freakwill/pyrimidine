#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from keras.models import Sequential, clone_model
from keras.layers import Dense

from pyrimidine import *
from pyrimidine.learn.base import BaseEstimator

import numpy.linalg as LA
import tensorflow as tf

def create_model(X, Y):
    hiden = 4
    model = Sequential()
    model.add(Dense(hiden, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(Y.shape[1], activation='softmax'))
    return model

def config(X, Y):
    _model = create_model(X, Y)
    
    class MyIndividual(MixedIndividual):
        element_class = FloatMatrixChromosome, FloatChromosome, FloatChromosome, FloatChromosome

        def _fitness(self):
            model = self.decode()
            return 1 / (LA.norm(model(X) - Y) +1)

        def decode(self):
            model = clone_model(_model)
            for k, layer in enumerate(model.layers):
                weights = (self.chromosomes[2*k], self.chromosomes[2*k+1])
                layer.set_weights(weights)
            return model

    class MyPopulation(HOFPopulation):
        element_class = MyIndividual

    pop = MyPopulation.random(n_individuals=40, sizes=((2,4),4,(4,2),2))
    return pop


def fit(X, y):
    pop = config(X,y)
    data=pop.evolve(n_iter=150, history=True)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Best Fitness', 'Mean Fitness']].plot(ax=ax)
    ax1 = ax.twinx()
    data[['STD Fitness']].plot(ax=ax)
    plt.show()
    return pop.best_.decode()

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,1], [1,0], [1,0], [0,1]])

model = create_model(X, Y)

# def f(model):
#     model.predict(X)

model=fit(X, Y)
print(model.get_weights())
print(model(X))


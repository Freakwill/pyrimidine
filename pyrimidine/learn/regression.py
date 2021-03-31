#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from digit_converter import IntervalConverter
c = IntervalConverter(lb=-60, ub=60)
class _BinaryChromosome(BinaryChromosome):
    def decode(self):
        return c(self)

c = IntervalConverter(lb=0, ub=5)
class _BinaryChromosome2(BinaryChromosome):
    def decode(self):
        return c(self)

import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import *
from pyrimidine.learn import BaseEstimator

class GALinearRegression(BaseEstimator, LinearRegression):
    '''Linear Regression

    solve Xp = y, with min_p ||Xp-y|| + a||p||, a>=0
    '''

    alpha = 0.05 # Regularization strength

    def postprocess(self):
        self.coef_ = self.best.chromosomes[0]
        self.intercept_ = self.best.chromosomes[1].decode()

    def config(self, X, y):
        class MyIndividual(SelfAdaptiveIndividual):
            params={'sigma':0.02}
            element_class = FloatChromosome, _BinaryChromosome, _BinaryChromosome2, FloatChromosome

            @property
            def sigma(self):
                return self.chromosomes[2].decode()

            def mutate(self, copy=False):
                self.fitness = None
                for chromosome in self.chromosomes[1:]:
                        chromosome.mutate()
                self.chromosomes[0].mutate(sigma=self.sigma)
                return self

            def _fitness(self):
                coef = self.chromosomes[0]
                intercept = self.chromosomes[1].decode()
                return - LA.norm(X @ coef +intercept - y) - GALinearRegression.alpha * LA.norm(coef, 1)

        class MyPopulation(SGA2Population):
            element_class = MyIndividual

        pop = MyPopulation.random(n_individuals=100, sizes=(11, 12, 10, 2))
        return pop


    def perf(self, n=10, *args, **kwargs):
        """Check the performance by running it several times
        
        Arguments:
            n {int} -- running times
        
        Returns:
            number -- mean time
        """
        import time
        times = []
        for _ in range(n):
            time1 = time.perf_counter()
            self.fit(*args, **kwargs)
            time2 = time.perf_counter()
            times.append(time2 - time1)
        return np.mean(times)

if __name__ == '__main__':
    def rel_error(y, t, m=None):
        if m is None:
            m = t.mean()
        return LA.norm(t - y) / LA.norm(t)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    data = pd.read_csv('~/Folders/Database/winequality.csv')
    keys = data.columns
    A = data[keys[:-1]].values # the rest is input
    B = data[keys[-1]].values  # the last colunm is ouput
    A, A_test, B, B_test = train_test_split(A, B, test_size=0.3)
    r = GALinearRegression()
    r.fit(A, B)
    print(f'''
coef_: {r.coef_}
intercept_: {r.intercept_}
train error: {r.score(A, B)}
test Error: {r.score(A_test, B_test)}''')
    r = LinearRegression()
    r.fit(A, B)
    print(f'''
coef_: {r.coef_}
intercept_: {r.intercept_}
train error: {r.score(A, B)}
test Error: {r.score(A_test, B_test)}''')

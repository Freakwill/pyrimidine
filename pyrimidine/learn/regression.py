#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from beagle import *
from digit_converter import BinaryConverter
bc = BinaryConverter(exponent=8)

import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import *

class GALinearRegression(LinearRegression):
    '''Linear Regression

    solve Xp = y, with min_p ||Xp-y|| + a||p||, a>=0
    '''

    alpha = 0.2 # Regularization strength

    def fit(self, X, y):
        pop = self.config(X,y)
        pop.evolve()
        best = pop.best_individual
        self.coef_ = best.chromosomes[0]
        self.intercept_ = bc(best.chromosomes[1])
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def config(self, X, y):
        class MyIndividual(MixIndividual):
            element_class = FloatChromosome, BinaryChromosome


            def _fitness(self):
                coef = self.chromosomes[0]
                intercept = bc(self.chromosomes[1])
                return - LA.norm(X @ coef +intercept - y) - GALinearRegression.alpha * LA.norm(coef, 1)

        class MyPopulation(SGAPopulation):
            element_class = MyIndividual

        pop = MyPopulation.random(n_individuals=40, sizes=(11, 12))
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
    print(f'''train error: {r.score(A, B)}
test Error: {r.score(A_test, B_test)}''')

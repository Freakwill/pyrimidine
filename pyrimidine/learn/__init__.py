#!/usr/bin/env python3

from sklearn.base import BaseEstimator as BE

import warnings, os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseEstimator(BE):

    estimated_params = ()

    @classmethod
    def config(cls, X, Y, *args, **kwargs):
        # configure a population for GA
        raise NotImplementedError

    def fit(self, X, Y, pop=None, warm_start=False):
        if warm_start:
            self.pop = pop or self.pop or self.config(X, Y)
        else:
            self.pop = pop or self.config(X, Y)
        self._fit(X, Y)
        return self

    def _fit(self, X, Y):
        self.pop.ezolve(n_iter=self.max_iter)
        model_ = self.pop.solution
        for k in self.estimated_params:
            setattr(self, k, getattr(model_, k))
#!/usr/bin/env python3

from sklearn.base import BaseEstimator as BE

import warnings, os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseEstimator(BE):

    """Base class for machine learning by GA
    
    Attributes:
        estimated_params (tuple): estimtated/learnable parameters by GA
        pop (Population): the population for GA
    """
    
    pop = None
    estimated_params = ()

    @classmethod
    def config(cls, X, Y=None, *args, **kwargs):
        # configure a population for GA
        raise NotImplementedError

    def fit(self, X, Y=None, pop=None, warm_start=False):
        if warm_start:
            self.pop = pop or self.pop or self.config(X, Y)
        else:
            self.pop = pop or self.config(X, Y)
        self._fit()
        return self

    def _fit(self):
        self.pop.ezolve(n_iter=self.max_iter)
        model_ = self.pop.solution
        for k in self.estimated_params:
            setattr(self, k, getattr(model_, k))


# def ga_opt(model_class):
#     class cls(BaseEstimator, model_class):
#         @classmethod
#         def create_model(cls, *args, **kwargs):
#             return model_class(*args, **kwargs)

#     return cls


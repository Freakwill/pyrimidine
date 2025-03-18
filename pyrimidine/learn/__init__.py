#!/usr/bin/env python3

# from .linear_regression import *
# from .cluster import *
# from .neural_network import *

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
        """Configure a population for GA based on the data X, Y;
        Subclasses must implement this method.
        
        Args:
            X (array): input data
            Y (array, optional): output data

        Returns:
            Population of GA
        """
        raise NotImplementedError

    def fit(self, X, Y=None, pop=None, warm_start=False):
        """fit method for the estimator
        
        Args:
            X (array): input data
            Y (array, optional): output data
            pop (None, optional): population for optimization
            warm_start (bool, optional): warm start or not
        
        Returns:
            the fitted estimator
        """

        if pop is None:
            if warm_start:
                if not hasattr(self, 'pop'):
                    self.pop = self.config(X, Y)
                    if hasattr(self, 'ind'):
                        self.pop.append(self.ind)
            else:
                self.pop = self.config(X, Y)
        else:
            self.pop = pop
        self._fit()
        return self

    def _fit(self):
        # the fit method for dirty work; get the solution after executing GA
        self.pop.ezolve(max_iter=self.max_iter)
        self.ind = self.pop.best_individual
        self.solution = self.pop.solution
        for k in self.estimated_params:
            setattr(self, k, getattr(self.solution, k))

    def predict(self, X):
        if not hasattr(self, 'solution'):
            raise 'Get the solution by GA first!'
        return self.solution.predict(X)

    def reset(self, include_ind=False):
        del self.pop
        if include_ind:
            del self.ind


# def ga_opt(model_class):
#     class cls(BaseEstimator, model_class):
#         @classmethod
#         def create_model(cls, *args, **kwargs):
#             return model_class(*args, **kwargs)

#     return cls


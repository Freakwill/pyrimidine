#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minin classes of iterative models
"""

import numpy as np
import pandas as pd

from ezstat import Statistics

from .utils import methodcaller, attrgetter
from .errors import *



class IterativeModel:
    # Mixin class for iterative algrithms

    _head = 'best solution & fitness'

    params = {'n_iter': 100}

    @property
    def solution(self):
        raise NotImplementedError('Have not defined `solution` for the model yet!')   

    @property
    def _row(self):
        best = self.solution
        return f'{best} & {best.fitness}'

    def init(self):
        pass
    

    def transit(self, *args, **kwargs):
        """
        The core method of the object.

        The method transitating one state of the object to another state,
        according to some rules, such as crossing and mutating for individuals in GA,
        or moving method in Simulated Annealing.
        """
        raise NotImplementedError('`transit`, the core of the algorithm, is not defined yet!')

    def local_search(self, *args, **kwargs):
        """
        The local search method for global search algorithm.
        """
        raise NotImplementedError('If you apply local search, then you have to define `local_search` method')

    def ezolve(self, n_iter=None, *args, **kwargs):
        # Extreamly eazy evolution method for lazybones
        n_iter = n_iter or self.n_iter
        self.init()
        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.postprocess()

    def evolve(self, n_iter=None, period=1, verbose=False, decode=False, stat={'Fitness': 'fitness'}, history=False, callbacks=()):
        """Get the history of the whole evolution

        Keyword Arguments:
            n_iter {number} -- number of iterations (default: {None})
            period {integer} -- the peroid of stat
            verbose {bool} -- to print the iteration process
            decode {bool} -- decode to the real solution
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            history {bool} -- True for recording history, or a DataFrame object recording previous history.
        
        Returns:
            DataFrame | None
        """

        n_iter = n_iter or self.n_iter
        self.init()

        if isinstance(stat, dict):
            stat = Statistics(stat)

        res = stat.do(self) if stat else {}
        keys = res.keys()

        if verbose:
            print(f"""
iteration & best solution & {" & ".join(res.keys())}
-------------------------------------------------------------
0 & {self.solution} & {" & ".join(map(str, res.values()))}""")

        if history is True:
            history = pd.DataFrame(data={k:[v] for k, v in res.items()})
            history_flag = True
        elif history is False:
            history_flag = False
        elif isinstance(history, pd.DataFrame):
            history_flag = True
        else:
            raise TypeError('Argument `history` should be an instance of pandas.DataFrame or bool.')
        # n_iter = n_iter or self.n_iter or self.default_n_iter
        for k in range(1, n_iter+1):
            self.transit(k)
            self.postprocess()
            for c in callbacks:
                c(self)
            res = stat.do(self) if stat else {}
            if history_flag and (period == 1 or k % period ==0):
                history = history.append(res, ignore_index=True)
            if verbose and (period == 1 or k % period ==0):
                print(f'{k} & {self.solution} & {" & ".join(str(res[k]) for k in keys)}')
        return history


    def get_history(self, *args, **kwargs):
        """Get the history of the whole evolution

        Would be replaced by `evolve`
        """
        raise DeprecationWarning('This method is deprecated from now on!!!, use `evolve(history=True, ***)` instead.')


    def perf(self, n_repeats=10, *args, **kwargs):
        """Get performance of Algo.

        Keyword Arguments:
            n_repeats {number} -- number of repeats to running algo. (default: {10})
        
        Returns:
            history, running time
        """
        import time
        times = []
        data = None
        for _ in range(n_repeats):
            cpy = self.clone(fitness=None)
            time1 = time.perf_counter()
            data0 = cpy.evolve(history=True, *args, **kwargs)
            time2 = time.perf_counter()
            times.append(time2 - time1)
            if data is None:
                data = data0
            else:
                data += data0
        return data / n_repeats, np.mean(times)

    def postprocess(self):
        pass

    def clone(self, type_=None, *args, **kwargs):
        raise NotImplementedError

    def encode(self):
        return self
 

    def save(self, filename='population.pkl'):
        import pickle
        if isinstance(filename, str):
            pklPath = pathlib.Path(filename)
        if pklPath.exists():
            print(Warning(f'There exists {filename}, It has been over written'))
        with open(pklPath, 'wb') as fo:
            pickle.dump(self, fo)

    @staticmethod
    def load(filename='population.pkl'):
        import pickle
        if isinstance(filename, str):
            pklPath = pathlib.Path('filename.pkl')
        if pklPath.exists():
            with open(pklPath, 'rb') as fo:
                return pickle.load(pklPath)
        else:
            raise IOError(f'Could not find {filename}!')


class FitnessModel(IterativeModel):
    """Iterative models with fitness

    The fitness should be stored until the the state of the model is changed.
    
    Extends:
        BaseIterativeModel
    
    Variables:
        __fitness {[type]} -- The value of a solution
    """

    __fitness = None

    @property
    def fitness(self):
        if self.__fitness is None:
            self.__fitness = self._fitness()
        return self.__fitness

    @fitness.setter
    def fitness(self, f):
        self.__fitness = f

    def get_fitness(self):
        return self._fitness()

    def _fitness(self):
        raise NotImplementedError

    def postprocess(self):
        self.__fitness = None

    @classmethod
    def set_fitness(cls, f=None):
        if f is None:
            if '_fitness' in globals():
                f = globals()['_fitness']
            else:
                raise Exception('Function `_fitness` is not defined before setting fitness. You may forget to create the class in the context of environment.')
        class C(cls):
            _fitness=f
        return C

    def clone(self, type_=None, fitness=None):
        if type_ is None:
            type_ = self.__class__
        if fitness is True:
            fitness = self.fitness
        return type_(list(map(methodcaller('clone', type_=type_.element_class, fitness=None), self)), fitness=fitness)

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Fitness':'fitness'}
 
        return super().evolve(stat=stat, *args, **kwargs)

    # def __rshift__(self, t):
    #     """
    #     Short for clone method
    #     """
    #     return self.clone(type_=t, fitness=True)

    # def __rlshift__(self, t):
    #     return self.clone(type_=t, fitness=True)


class PopulationModel(FitnessModel):
    """subclass of BaseFitnessModel

    It is consisted of a set of solutions.
    """

    __sorted_individuals = []

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Best Fitness':'best_fitness', 'Mean Fitness':'mean_fitness', 'STD Fitness':'std_fitness', 'Population': 'n_elements'}
        return super().evolve(stat=stat, *args, **kwargs)

    @property
    def individuals(self):
        return self.__elements

    @individuals.setter
    def individuals(self, x):
        # Set the fitness to be None, when setting individuals of the object
        self.__elements = x
        self.n_individuals = len(x)
        self.sorted = False
        self.fitness = None

    @property
    def n_individuals(self):
        return self.n_elements

    @n_individuals.setter
    def n_individuals(self, v):
        self.__n_elements = v

    def _fitness(self):
        """Calculate the fitness of the whole population

        Fitness of a population is the average fitness by default.
        """
        return self.mean_fitness

    def _fitnesses(self):
        return list(map(attrgetter('fitness'), self.individuals))

    @property
    def mean_fitness(self):
        return np.mean(self._fitnesses())

    @property
    def std_fitness(self):
        return np.std(self._fitnesses())

    @property
    def best_fitness(self):
        return np.max(self._fitnesses())

    @property
    def fitnesses(self):
        return np.array(self._fitnesses())

    def get_best(self, key='fitness'):
        # Get best individual under `key`
        k = np.argmax(tuple(map(attrgetter(key), self.individuals)))
        return self.individuals[k]

    def get_best_individuals(self, n=1):
        # first n best individuals
        if n < 1:
            n = int(self.n_individuals * n)
        elif not isinstance(n, int):
            n = int(n)
        return self.sorted_individuals[-n:]

    def get_worst(self, key='fitness'):
        k = np.argmin(tuple(map(attrgetter(key), self.individuals)))
        return self.individuals[k]

    # Following is some useful aliases
    @property
    def worst_individual(self):
        k = np.argmin(self._fitnesses())
        return self.individuals[k]

    @property
    def best_(self):
        return self.best_individual

    @property
    def best_individual(self):
        k = np.argmax(self._fitnesses())
        return self.individuals[k]

    @property
    def solution(self):
        return self.best_individual

    @property
    def sorted_individuals(self):
        if self.__sorted_individuals == []:
            ks = np.argsort(self._fitnesses())
            self.__sorted_individuals = [self.individuals[k] for k in ks]
        return self.__sorted_individuals

    @sorted_individuals.setter
    def sorted_individuals(self, s):
        self.__sorted_individuals = s

    def sort(self):
        # sort the whole population
        ks = self.argsort()
        self.individuals = [self.individuals[k] for k in ks]


    def argsort(self):
        return np.argsort(self._fitnesses())


    def drop(self, n=1):
        if n < 1:
            n = int(self.n_individuals * n)
        elif not isinstance(n, int):
            n = int(n)
        ks = self.argsort()
        self.individuals = [self.individuals[k] for k in ks[n:]]

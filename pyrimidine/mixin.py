#!/usr/bin/env python3


"""
Mixin classes of iterative models

IterativeModel: base class of all iterative models
FitnessModel: IterativeModel with `fitness`
ContainerModel: base class of all iterative models with multi-objects
PopulationModel: subclass of FitnessModel, population-like iterative models

Relation of the classes:
IterativeModel  --->  ContainerModel
    |                      |
    |                      |
    v                      v
FitnessModel  --->  PopulationModel
"""


import numpy as np
import pandas as pd

from ezstat import Statistics

from operator import methodcaller, attrgetter
from .errors import *


class IterativeModel:
    # Mixin class for iterative algrithms
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


    def transition(self, *args, **kwargs):
        """
        The core method of the object.

        The method transitating one state of the object to another state,
        according to some rules, such as crossing and mutating for individuals in GA,
        or moving method in Simulated Annealing.
        """
        raise NotImplementedError('`transition`, the core of the algorithm, is not defined yet!')


    def local_search(self, *args, **kwargs):
        """
        The local search method for global search algorithm.
        """
        raise NotImplementedError('If you apply local search, then you have to define `local_search` method')


    def ezolve(self, n_iter=None):
        # Extreamly eazy evolution method for lazybones
        n_iter = n_iter or self.n_iter
        self.init()
        for k in range(1, n_iter+1):
            self.transition(k)


    def evolve(self, n_iter:int=100, period:int=1, verbose:bool=False, decode=False, history=False, stat=None, attrs=('state',), control=None):
        """Get the history of the whole evolution

        Keyword Arguments:
            n_iter {number} -- number of iterations (default: {None})
            period {integer} -- the peroid of stat
            verbose {bool} -- to print the iteration process
            decode {bool} -- decode to the real solution
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            history {bool|DataFrame} -- True for recording history, or a DataFrame object recording previous history.
        
        Returns:
            DataFrame | None
        """
        assert control is None or callable(control)
 
        n_iter = n_iter or self.n_iter

        if isinstance(stat, dict): stat = Statistics(stat)

        self.init()

        if verbose:
            res = stat(self) if stat else {}
            print(f"""
iteration & {" & ".join(attrs)} & {" & ".join(res.keys())}
-------------------------------------------------------------
[0] & {" & ".join(str(getattr(self, attr)) for attr in attrs)} & {" & ".join(map(str, res.values()))}""")

        if history is True:
            res = stat(self) if stat else {}
            history = pd.DataFrame(data={k:[v] for k, v in res.items()})
            history_flag = True
        elif history is False:
            history_flag = False
        elif isinstance(history, pd.DataFrame):
            history_flag = True
        else:
            raise TypeError('Argument `history` should be an instance of pandas.DataFrame or boolean.')
        # n_iter = n_iter or self.n_iter
        for k in range(1, n_iter+1):
            self.transition(k)
            if history_flag and (period == 1 or k % period ==0):
                res = stat(self) if stat else {}
                history = pd.concat([history,
                    pd.Series(res.values(), index=res.keys()).to_frame().T],
                    ignore_index=True)
            if verbose and (period == 1 or k % period ==0):
                print(f'[{k}] & {" & ".join(str(getattr(self, attr)) for attr in attrs)} & {" & ".join(map(str, res.values()))}')
            
            if control:
                if control(self):
                    break
        return history


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
            data0 = cpy.evolve(verbose=False, *args, **kwargs)
            time2 = time.perf_counter()
            times.append(time2 - time1)
            if data is None:
                data = data0
            else:
                data += data0
        return data / n_repeats, np.mean(times)


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


    def _fitness(self):
        raise NotImplementedError

    def get_fitness(self):
        # the alias of the fitness
        return self._fitness()


    @classmethod
    def set_fitness(cls, f=None):
        if f is None:
            if '_fitness' in globals():
                f = globals()['_fitness']
            else:
                raise Exception('Function `_fitness` is not defined before setting fitness. You may forget to create the class in the context of environment.')
        class C(cls):
            def _fitness(self):
                return f(self)
        return C


    def clone(self, type_=None, fitness=None):
        if type_ is None:
            type_ = self.__class__
        if fitness is True:
            fitness = self.fitness
        cpy = type_(list(map(methodcaller('clone', type_=type_.element_class, fitness=None), self)))
        if fitness is True:
            cpy.__fitness = self.fitness
        else:
            cpy.__fitness = fitness
        return cpy


    def evolve(self, stat=None, attrs=('solution',), *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Fitness':'fitness'}

        return super().evolve(stat=stat, attrs=attrs, *args, **kwargs)

    def after_setter(self):
        # clean up the fitness after updating the chromosome
        self.__fitness = None


class ContainerModel(IterativeModel):
    def init(self, *args, **kwargs):
        for element in self:
            element.init(*args, **kwargs)

    def transition(self, *args, **kwargs):
        for element in self:
            element.transition(*args, **kwargs)


class PopulationModel(FitnessModel, ContainerModel):
    """mixin class for population-based heuristic algorithm

    It is consisted of a set of solutions.
    """

    __sorted_ = []

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Best Fitness':'best_fitness', 'Mean Fitness':'mean_fitness', 'STD Fitness':'std_fitness', 'Population': 'n_elements'}
        return super().evolve(stat=stat, *args, **kwargs)


    def after_setter(self):
        # self.sorted = False
        self.__sorted_ = []
        self.__fitness = None


    def _fitness(self):
        """Calculate the fitness of the whole population

        Fitness of a population is the best fitness by default.
        """
        return self.best_fitness


    def get_fitnesses(self):
        return list(map(attrgetter('fitness'), self))


    @property
    def mean_fitness(self):
        return np.mean(self.get_fitnesses())


    @property
    def std_fitness(self):
        return np.std(self.get_fitnesses())


    @property
    def best_fitness(self):
        return np.max(self.get_fitnesses())


    @property
    def fitnesses(self):
        return np.array(self.get_fitnesses())


    def get_best_individual(self, key='fitness', copy=False):
        # Get best individual under `key`
        k = np.argmax(tuple(map(attrgetter(key), self)))
        if copy:
            return self[k].clone()
        else:
            return self[k]


    def get_best_individuals(self, n=1, copy=False):
        # Get first n best individuals
        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)

        if copy:
            return [a.clone() for a in self.sorted_[-n:]]
        else:
            return self.sorted_[-n:]


    def get_worst(self, key='fitness', copy=False):
        k = np.argmin(tuple(map(attrgetter(key), self)))
        if copy:
            return self[k].clone()
        else:
            return self[k]


    # Following is some useful aliases
    @property
    def worst_individual(self):
        k = np.argmin(self.get_fitnesses())
        return self[k]


    @property
    def best_(self):
        return self.best_individual


    @property
    def best_individual(self):
        k = np.argmax(self.get_fitnesses())
        return self.individuals[k]


    @property
    def solution(self):
        return self.best_individual


    @property
    def sorted_(self):
        if self.__sorted_ == []:
            ks = self.argsort()
            self.__sorted_ = [self[k] for k in ks]
        return self.__sorted_


    @sorted_.setter
    def sorted_(self, s):
        self.__sorted_ = s


    def sort(self):
        # sort the whole population
        ks = self.argsort()
        self.elements = [self[k] for k in ks]


    def argsort(self):
        return np.argsort(self.get_fitnesses())


    def drop(self, n=1):
        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)
        ks = self.argsort()
        self.elements = [self[k] for k in ks[n:]]

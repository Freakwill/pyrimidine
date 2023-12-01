#!/usr/bin/env python3


"""
Mixin classes for iterative algorithms or models

IterativeMixin: base class for all iterative algorithms
FitnessMixin: IterativeMixin with `fitness`
CollectiveMixin: base class for all iterative algorithms with multi-objects
PopulationMixin: subclass of FitnessMixin, population-like iterative algorithms

Relation of the classes:
IterativeMixin  --->  CollectiveMixin
    |                      |
    |                      |
    v                      v
FitnessMixin  --->  PopulationMixin
"""


from operator import methodcaller, attrgetter

import numpy as np
import pandas as pd

from ezstat import Statistics
from .errors import *

from .deco import side_effect


class IterativeMixin:
    # Mixin class for iterative algrithms

    params = {'n_iter': 100}

    # @property
    # def _row(self):
    #     best = self.solution
    #     return f'{best} & {best.fitness}'

    def init(self):
        pass

    def transition(self, *args, **kwargs):
        """The core method of the object.

        This method transitions one state of the object to another state
        based on certain rules, such as crossing and mutating for individuals in GA,
        or the moving method in Simulated Annealing.
        """
        raise NotImplementedError('`transition` (the core of the algorithm) is not defined yet!')

    def local_search(self, *args, **kwargs):
        """
        The local search method for a global search algorithm.
        """
        raise NotImplementedError('If you apply a local search algorithm, you must define the `local_search` method.')

    def ezolve(self, n_iter=None):
        # Extreamly eazy evolution method for lazybones
        n_iter = n_iter or self.n_iter
        self.init()
        for k in range(1, n_iter+1):
            self.transition(k)

    def evolve(self, n_iter:int=100, period:int=1, verbose:bool=False, decode=False, history=False, stat=None, attrs=('solution',), control=None):
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
            raise TypeError('The argument `history` should be an instance of `pandas.DataFrame` or `bool`.')
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
            cpy = self.clone(cache=False)
            data0 = cpy.evolve(verbose=False, *args, **kwargs)
            if data is None:
                data = data0
            else:
                data += data0
        return data / n_repeats

    def clone(self, type_=None, *args, **kwargs):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def decode(self):
        return self

    @property
    def solution(self):
        return self.decode()
 
    def save(self, filename='model.pkl', check=False):
        import pickle
        if isinstance(filename, str):
            pklPath = pathlib.Path(filename)
        if check and pklPath.exists():
            raise FileExistsError(f'File {filename} has exist!')
        with open(pklPath, 'wb') as fo:
            pickle.dump(self, fo)

    @staticmethod
    def load(filename='model.pkl'):
        import pickle
        if isinstance(filename, str):
            pklPath = pathlib.Path('filename.pkl')
        if pklPath.exists():
            with open(pklPath, 'rb') as fo:
                return pickle.load(pklPath)
        else:
            raise FileNotFoundError(f'Could not find the file {filename}!')

    def after_setter(self):
        if hasattr(self, '_cache'):
            self.clear_cache()


class FitnessMixin(IterativeMixin):
    """Iterative models drived by the fitness/objective function

    The fitness should be stored until the the state of the model is changed.
    
    Extends:
        IterativeMixin
    """

    def get_fitness(self):
        raise NotImplementedError

    def _fitness(self):
        raise NotImplementedError

    @property
    def fitness(self):
        return self._fitness()

    @classmethod
    def set_fitness(cls, f=None):
        if f is None:
            if '_fitness' in globals():
                f = globals()['_fitness']
            else:
                raise Exception('Function `_fitness` is not defined before setting fitness. You may forget to create the class in the context of environment.')
        cls._fitness = f
        return cls

    def evolve(self, stat=None, attrs=('solution',), *args, **kwargs):
        """Get the history of solution and its fitness by default.
        """

        if stat is None:
            stat = {'Fitness': 'fitness'}
        return super().evolve(stat=stat, attrs=attrs, *args, **kwargs)


class CollectiveMixin(IterativeMixin):

    map = map

    def init(self, *args, **kwargs):
        for element in self:
            element.init(*args, **kwargs)

    def transition(self, *args, **kwargs):
        for element in self:
            element.transition(*args, **kwargs)

    @side_effect
    def remove(self, individual):
        self.elements.remove(individual)

    @side_effect
    def pop(self, k=-1):
        self.elements.pop(k)

    @side_effect
    def extend(self, inds):
        self.elements.extend(inds)

    @side_effect
    def append(self, ind):
        self.elements.append(ind)


class PopulationMixin(FitnessMixin, CollectiveMixin):
    """mixin class for population-based heuristic algorithm

    It is consisted of a collection of solutions.
    """

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """

        if stat is None:
            stat = {'Best Fitness': 'best_fitness', 'Mean Fitness': 'mean_fitness',
            'STD Fitness': 'std_fitness', 'Population': 'n_elements'}
        return super().evolve(stat=stat, *args, **kwargs)

    @classmethod
    def set_fitness(cls, *args, **kwargs):
        # set fitness for the element_class.
        cls.element_class.set_fitness(*args, **kwargs)
        return cls

    @property
    def fitness(self):
        return self.best_fitness

    def _fitness(self):
        """Calculate the fitness of the whole population

        Fitness of a population is the best fitness by default.
        (not recommended to be overridden)
        """
        return self.best_fitness

    def get_all_fitness(self):
        return list(self.map(attrgetter('fitness'), self))

    def get_all(self, key='fitness'):
        return list(self.map(attrgetter(key)), self)

    @property
    def mean_fitness(self):
        return np.mean(self.get_all_fitness())

    @property
    def std_fitness(self):
        return np.std(self.get_all_fitness())

    @property
    def best_fitness(self):
        return np.max(self.get_all_fitness())

    @property
    def stat_fitness(self, s=np.max):
        f = self.get_all_fitness()
        if isinstance(s, tuple):
            return tuple(si(f) for si in  s)
        else:
            return s(f)

    def get_best_element(self, copy=False):
        """Get best element
        
        Args:
            copy (bool, optional): return the copy of the selected element, if `copy=True`
        
        Returns:
            An element
        """

        k = np.argmax(self.get_all_fitness())
        if copy:
            return self[k].clone()
        else:
            return self[k]

    def get_best_elements(self, n=1, copy=False):
        """Get first n best elements
        
        Args:
            n (int, optional): the number of elements selected
            copy (bool, optional): if copy=True, then it returns the copies of elements
        
        Returns:
            n elements
        """

        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)

        if copy:
            return [self[k].clone() for k in self.argsort()[-n:]]
        else:
            return [self[k] for k in self.argsort()[-n:]]

    @property
    def best_element(self):
        """Get the best element

        The difference between `best_element` and `get_best_element` is that
        `best_element` only returns the reference of the selected element.
        
        Returns:
            The best element
        """

        k = np.argmax(self.get_all_fitness())
        return self[k]

    @property
    def solution(self):
        return self.best_element.decode()

    def get_worst_element(self, copy=False):
        k = np.argmin(self.get_all_fitness())
        if copy:
            return self[k].clone()
        else:
            return self[k]

    @property
    def worst_element(self):
        k = np.argmin(self.get_all_fitness())
        return self[k]

    def sorted_(self):
        # return a list of sorted individuals
        return [self[k] for k in self.argsort()]

    def sort(self):
        # sort the whole population
        ks = self.argsort()
        self.__elements = [self[k] for k in ks]

    def argsort(self):
        return np.argsort(self.get_all_fitness())

    def drop(self, n=1):
        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)
        ks = self.argsort()
        self.elements = [self[k] for k in ks[n:]]

    def clone(self, type_=None, element_class=None, *args, **kwargs):
        if type_ is None:
            type_ = self.__class__
        cpy = type_(list(map(methodcaller('clone', type_=element_class or type_.element_class), self)))
        return cpy

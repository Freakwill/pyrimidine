#!/usr/bin/env python3


"""
Mixin classes for iterative algorithms or models,
where the basic operations are defined.

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


from itertools import chain
from operator import methodcaller, attrgetter

import numpy as np
import pandas as pd

try:
    from ezstat import Statistics
except:
    from ._stat import Statistics

from .deco import side_effect

from .errors import *


class IterativeMixin:
    # Mixin class for iterative algrithms

    params = {'max_iter': 100}

    # @property
    # def _row(self):
    #     best = self.solution
    #     return f'{best} & {best.fitness}'

    def init(self):
        # initalize the object
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

    def ezolve(self, max_iter=None, initialize=True):
        """Extremely eazy evolution method for lazybones

        Keyword Arguments:
            initialize {bool} -- set True to call `init` method (default: {True})
            max_iter {number} -- number of iterations (default: {None})
        """
        max_iter = max_iter or self.max_iter
        if initialize:
            self.init()
        for k in range(1, max_iter+1):
            self.transition(k)

    def evolve(self, initialize:bool=True, max_iter:int=100, period:int=1, verbose:bool=False, history=False, stat=None, attrs=('solution',), control=None):
        """Let the population evolve automatically.

        To get the history of the whole evolution via setting `history=True`

        Keyword Arguments:
            initialize {bool} -- set True to call `init` method (default: {True})
            max_iter {number} -- number of iterations (default: {None})
            period {integer} -- the peroid of stat (default: {1})
            verbose {bool} -- to print the iteration process (default: {False})
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            history {bool|DataFrame} -- True for recording history, or a DataFrame object recording previous history.  (default: {False})
            attrs {tuple[str]} -- attributes of the object
            control -- control information for the iteration
        
        Returns:
            DataFrame | None
        """

        assert control is None or callable(control)
 
        max_iter = max_iter or self.max_iter

        if isinstance(stat, dict): stat = Statistics(stat)
        
        if initialize:
            self.init()

        if history is True:
            res = stat(self)
            history = pd.DataFrame(data={k:[v] for k, v in res.items()})
            history_flag = True
        elif history is False:
            history_flag = False
        elif isinstance(history, pd.DataFrame):
            history_flag = True
        else:
            raise TypeError('The argument `history` should be an instance of `pandas.DataFrame` or `bool`.')

        if verbose:
            def _row(t, attrs, res, sep=" & "):
                return sep.join(map(str, chain(("[%d]"%t,), (getattr(self, attr) for attr in attrs), res.values())))
            if not history_flag:
                res = stat(self)
            print(f"""            ** History **
{" & ".join(chain(("iteration",), attrs, res.keys()))}
-------------------------------------------------------------""")
            print(_row(0, attrs, res))

        for t in range(1, max_iter+1):
            self.transition(t)

            if history_flag and (period == 1 or t % period ==0):
                res = stat(self)
                history = pd.concat([history,
                    pd.Series(res.values(), index=res.keys()).to_frame().T],
                    ignore_index=True)

            if verbose and (period == 1 or t % period ==0):
                if not history_flag:
                    res = stat(self)
                print(_row(t, attrs, res))

            if control:
                if control(self):
                    # if it satisfies the control condition (such as convergence criterion)
                    break

        return history

    def perf(self, n_repeats=10, timing=True, *args, **kwargs):
        """Get performance of Algo.

        Keyword Arguments:
            n_repeats {number} -- number of repeats to running algo. (default: {10})
            timing {bool} -- measure execution the time
        
        Returns:
            history, time
        """

        from time import process_time

        times = []
        data = None
        for _ in range(n_repeats):
            cpy = self.copy(cache=False)
            data0 = cpy.evolve(verbose=False, *args, **kwargs)
            if timing:
                start = process_time()
                data0 = cpy.evolve(verbose=False, *args, **kwargs)
                end = process_time()
                delta = end - start
            times.append(delta)
            if data is None:
                data = data0
            else:
                data += data0

        return data / n_repeats, np.mean(times)

    def copy(self, *args, **kwargs):
        raise NotImplementedError

    def clone(self):
        return self.copy()

    def decode(self):
        return self

    @classmethod
    def encode(cls, x):
        # encode x to a chromosome
        raise NotImplementedError

    @property
    def solution(self):
        # get the real solution
        return self.decode()
 
    def save(self, filename=None, check=False):
        """Save the object to file using pickle
        
        Args:
            filename (None, optional): the path of the pickle file
            check (bool, optional): check whether the file has existed.
        
        Raises:
            FileExistsError
        """
        import pickle, pathlib
        if filename is None:
            filename = f'{self.__class__.__name__}-{self.__name__}'
        if isinstance(filename, str):
            pklPath = pathlib.Path(filename).with_suffix('.pkl')
        if check and pklPath.exists():
            raise FileExistsError(f'File {filename} has existed!')
        with open(pklPath, 'wb') as fo:
            pickle.dump(self, fo)

    @classmethod
    def load(cls, filename=None):
        """Load the object from the pickle file
        
        Args:
            filename (None, optional): the path of the pickle file
        
        Returns:
            TYPE: the object of `cls`

        Raises:
            FileNotFoundError: Do not find the file
        """
        import pickle, pathlib
        if filename is None:
            filename = f'{cls.__name__}-{pop}'
        if isinstance(filename, str):
            pklPath = pathlib.Path(filename).with_suffix('.pkl')
        if pklPath.exists():
            with open(pklPath, 'rb') as fo:
                obj = pickle.load(fo)
                if not isinstance(obj, cls):
                    raise TypeError('The object loaded is not an instance of "{cls}".')
        else:
            raise FileNotFoundError(f'Could not find the file {filename}!')

    def after_setter(self):
        # what should be done after setting the attributes of the object
        if hasattr(self, '_cache'):
            self.clear_cache()

    @classmethod
    def solve(cls, *args, **kwargs):
        # get the solution after evolution immediately
        return cls.random().ezolve(*args, **kwargs).solution


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
    def set_fitness(cls, f=None, decode=None):
        """Set fitness computing method
        
        Args:
            f (None, optional): function to evalute the fintess
            decode (None, optional): decode the individual or not before calcuating fitness
                                    (deprecated currently, eps. when you use `memory`!)
        
        Returns:
            the class with the fitness `f`
        """

        if f is None:
            if '_fitness' in globals():
                f = globals()['_fitness']
            else:
                raise Exception('Function `_fitness` is not defined before setting fitness. You may forget to create the class in the context of environment.')
        if not decode:
            def _fitness(obj):
                return f(obj)
        else:
            def _fitness(obj):
                return f(obj.decode())
        cls._fitness = _fitness
        return cls

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of solution and its fitness by default.
        """

        if stat is None:
            stat = {'Fitness': 'fitness'}
        return super().evolve(stat=stat, *args, **kwargs)

    def copy(self, type_=None, element_class=None, *args, **kwargs):
        """copy the object
        
        Args:
            type_: the type of new object
            element_class (None, optional): the new element_class
        
        Returns:
            a new object copy the data but with new type
        """
        type_ = type_ or self.__class__
        element_class = element_class or type_.element_class
        if isinstance(type_.element_class, tuple):
            return type_([c.copy(type_=t) for c, t in zip(self, type_.element_class)])
        else:
            return type_([c.copy(type_=element_class) for c in self])

    def clone(self):
        # totally copy the object
        return self.__class__(list(map(methodcaller('clone'), self)))

    # def diff_fitness(self):
    #     return self.fitness - self.previous_fitness

    # def previous_fitness(self):
    #     return self.memory['fitness']


class CollectiveMixin(IterativeMixin):
    # mixin class for swarm intelligent algorithm

    map = map  # use the biuld-in `map` function by default

    def init(self, *args, **kwargs):
        for element in self:
            if hasattr(element, 'init'):
                element.init(*args, **kwargs)

    def transition(self, *args, **kwargs):
        for element in self:
            element.transition(*args, **kwargs)

    @side_effect
    def remove(self, element):
        """remove an element from the list-like object
        
        Args:
            element: an element in the object
        
        Returns:
            CollectiveMixin
        """
        self.elements.remove(element)

    @side_effect
    def pop(self, k=-1):
        """pop an element from the list-like object
        
        Args:
            element: an element in the object
        
        Returns:
            CollectiveMixin
        """
        self.elements.pop(k)

    @side_effect
    def extend(self, elements):
        """extend the list-like object by elements
        
        Args:
            elements: a list of elements
        
        Returns:
            CollectiveMixin
        """
        self.elements.extend(elements)

    @side_effect
    def append(self, element):
        """append the list-like object with an element
        
        Args:
            element: an element
        
        Returns:
            CollectiveMixin
        """

        self.elements.append(element)

    def random_select(self, n_sel=None, copy=False):
        """append the list-like object with an element
        
        Args:
            n_sel (int): the number of elements selected
            copy (bool): copy the element or not
        
        Returns:
            An element (or its copy)
        """

        if n_sel is None:
            k = np.random.randint(len(self))
            return self[k]
        else:
            ks = np.random.randint(len(self), size=n_sel)
            return self[ks]

    def observe(self, name='_system'):
        """for observer design pattern
        Args:
            name (str): the attribute for the algebra system

        Returns:
            An element (or its copy)
        """

        for o in self:
            setattr(o, name, self)

    @property
    def op(self):
        """operators for algebraic-probramming

        Example:
           a.op['x'](b) == self._system.x(a, b)
           # where `x` is an algebraic operation defined in `_system`
        """

        class _C:
            def __getitem__(obj, s):
                def _f(*args, **kwargs):
                    return getattr(self._system, s)(self, *args, **kwargs)
                return _f
        return _C()


class PopulationMixin(FitnessMixin, CollectiveMixin):
    """mixin class for population-based heuristic algorithm

    It is consisted of a collection of solutions.
    """

    def evolve(self, stat=None, *args, **kwargs):
        """To get the history of the whole evolution
        """

        if stat is None:
            stat = {'Max Fitness': 'max_fitness', 'Mean Fitness': 'mean_fitness',
            'STD Fitness': 'std_fitness'}
        return super().evolve(stat=stat, *args, **kwargs)

    @classmethod
    def set_fitness(cls, *args, **kwargs):
        # To set fitness for the element_class
        if hasattr(cls.element_class, 'set_fitness'):
            cls.element_class.set_fitness(*args, **kwargs)
        else:
            raise AttributeError(f'{cls.element_class} does not have `set_fitness`')
        return cls

    @property
    def fitness(self):
        # The fitness of the entire population is the maximum fitness of the individuals
        return self.max_fitness

    # def _fitness(self):
    #     """Calculate the fitness of the whole population

    #     Fitness of a population is the best fitness by default.
    #     (not recommended to be overridden)
    #     """
    #     return self.max_fitness

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
        DeprecationWarning('`best_fitness` is depricated and please use `max_fitness`')
        return np.max(self.get_all_fitness())

    @property
    def max_fitness(self):
        return np.max(self.get_all_fitness())

    @property
    def stat_fitness(self, s=np.max):
        """Do statistics for the fitnesses of individuals in the population
        
        Args:
            s (a function or a tuple of functions): a statistic
        
        Returns:
            number or a tuple of numbers
        """
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
            An element: the element with max fitness
        """

        k = np.argmax(self.get_all_fitness())
        if copy:
            return self[k].copy()
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
            return [self[k].copy() for k in self.argsort()[-n:]]
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
        # see get_best_element
        k = np.argmin(self.get_all_fitness())
        if copy:
            return self[k].copy()
        else:
            return self[k]

    def get_worst_elements(self, n=1, copy=False):
        # see get_best_elements
        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)

        if copy:
            return [self[k].copy() for k in self.argsort()[:n]]
        else:
            return [self[k] for k in self.argsort()[:n]]

    @property
    def worst_element(self):
        # like best_element
        k = np.argmin(self.get_all_fitness())
        return self[k]

    def get_samples(self, copy=False, *args, **kwargs):
        # get a part of the population randomly

        if copy:
            return [self[k].copy() for k in np.random.choice(np.arange(len(self)), *args, **kwargs)]
        else:
            return [self[k] for k in np.random.choice(np.arange(len(self)), *args, **kwargs)]

    def sorted_(self):
        """a list of sorted individuals

        Returns:
            list: the list of elements after sorting
        """

        return [self[k] for k in self.argsort()]

    def sort(self):
        # sort the whole population
        ks = self.argsort()
        self.__elements = self[ks]

    def argsort(self):
        # get the index after sorting the whole population
        return np.argsort(self.get_all_fitness())

    def drop(self, n=1):
        # drop the worst n elements
        if n < 1:
            n = int(self.n_elements * n)
        elif not isinstance(n, int):
            n = int(n)
        ks = self.argsort()
        self.elements = [self[k] for k in ks[n:]]


class MultiPopulationMixin(PopulationMixin):

    @property
    def solution(self):
        return self.best_element.solution()


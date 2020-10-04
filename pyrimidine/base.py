#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
the main module of pyrimidine

main classes:
BaseGene: the gene of chromosome
BaseChromosome: sequence of genes, represents part of a solution
BaseIndividual: sequence of chromosomes, represents a solution of a problem
BasePopulation: set of individuals, represents a set of a problem
                also the state of a stachostic process
BaseSpecies: set of population for more complicated optimalization


Subclass the classes and override some main method esp. _fitness.

Example:
    select ti, ni from t, n
    sum of ni ~ 10, while ti dose not repeat

The opt. problem is
    min sum of {ni} and maximum of frequences in {ti}
    where i is selected.

t = np.random.randint(1, 5, 100)
n = np.random.randint(1, 4, 100)

import collections
def max_repeat(x):
    # maximum of numbers of repeats
    c = collections.Counter(x)
    bm=np.argmax([b for a, b in c.items()])
    return list(c.keys())[bm]

class MyIndividual(MonoIndividual):

    element_class = BinaryChromosome

    def _fitness(self):
        x = self.evaluate()
        return - x[0] - x[1]

    def evaluate(self):
        return np.dot(n, self[0]), max_repeat(ti for ti, c in zip(t, self.chromosomes[0]) if c==1)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=50, size=100)
pop.evolve()
print(pop.best_individual)
"""

import types
from operator import attrgetter
from random import random, choice
import numpy as np
from .utils import uniformly_select
from .errors import *
from .meta import MetaHighContainer, MetaContainer, MetaTuple, MetaList


class BaseIterativeModel:

    goal_function = None

    _head = 'best solution & fitness'

    params = {}

    # def __getitem__(self, key):
    #     return self.params[key]

    # def __setitem__(self, key, value):
    #     self.params[key] = value
    # 
    
    def __new__(cls, *args, **kwargs):
        # constructor of BaseIterativeModel
        obj = super(BaseIterativeModel, cls).__new__(cls)
        for k, v in cls.params.items():
            setattr(obj, k, v)
        return obj

    # def __getattr__(self, key):
    #     return self.params[key]
    #     
    
    # def config(self, params, **kwargs):
    #     self.params.update(params)
    #     for k, v in kwargs.items():
    #         setter(self, k, v)

    @property
    def _row(self):
        best = self.solution
        return f'{best} & {best.fitness}'

    def init(self):
        pass
    

    def transitate(self, *args, **kwargs):
        """
        The core method of the object.

        The method transitating one state of the object to another state,
        according to some rules, such as crossing and mutating for individuals in GA,
        or moving method in Simulated Annealing.
        """
        raise NotImplementedError

    def local_search(self, *args, **kwargs):
        """
        The local search method for global search algorithm.
        """
        raise NotImplementedError

    def evolve(self, n_iter=100, per=1, verbose=False, decode=False, *args, **kwargs):
        if verbose:
            print('iteration & ' , self._head)
            print('-------------------------------------------------------------')
            print('0 & ', self._row)
        self.init()
        # n_iter = n_iter or self.n_iter or self.default_n_iter
        for k in range(1, n_iter+1):
            self.transitate(k, *args, **kwargs)
            self.post_process()
            if verbose and (per == 1 or k % per ==0):
                print(f'{k} & ', self._row)

    def history(self, n_iter=100, stat=None, *args, **kwargs):
        """Get the history of the whole evolution

        Keyword Arguments:
            n_iter {number} -- number of iterations (default: {100})
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            (default: {'Fitness': 'fitness'})
        
        Returns:
            DataFrame
        """
        if stat is None:
            stat = {'Fitness':'fitness'}
 
        self.init()
        import pandas as pd
        H = pd.DataFrame(data={k:[(getattr(self, s)() if isinstance(getattr(self, s), types.FunctionType) else getattr(self, s)) if isinstance(s, str) and hasattr(self, s) else s(self)] for k, s in stat.items()})
        for k in range(n_iter):
            self.transitate(k, *args, **kwargs)
            self.post_process()
            H = H.append({k:(getattr(self, s)() if isinstance(getattr(self, s), types.FunctionType) else getattr(self, s)) if isinstance(s, str) and hasattr(self, s) else s(self) for k, s in stat.items()}, ignore_index=True)
        return H

    def perf(self, n_repeat=10, *args, **kwargs):
        import time
        times = []
        data = None
        for _ in range(n_repeat):   
            time1 = time.perf_counter()
            data0 = self.history(*args, **kwargs)
            time2 = time.perf_counter()
            times.append(time2 - time1)
            if data is None:
                data = data0
            else:
                data += data0
        return data / n_repeat, np.mean(times)

    def post_process(self):
        pass

    # @classmethod
    # def redefine(cls, m, f):
    #     import types
    #     cls_=types.new_class(cls.__name__, (cls,), {m: f})
    #     return cls_


class Solution(object):
    '''[Summary for Class Solution]'''
    def __init__(self, value, goal_value=None):
        self.value = value
        self.goal_value = goal_value

    def __str__(self):
        if self.goal_value is None:
            return ' | '.join(str(x) for x in self.value)
        else:
            return f"{' | '.join(str(x) for x in self.value)} & {self.goal_value}"


class BaseGene:
    values = ()

    def __repr__(self):
        return self.__class__.__name__ + f': {self}'

    def __str__(self):
        return f'{self}'

    @classmethod
    def _random(cls):
        return cls(choice(cls.values))

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(np.random.choice(cls.values, *args, **kwargs))


class BaseFitnessModel(BaseIterativeModel):
    __fitness = None

    @property
    def fitness(self):
        if self.__fitness is None:
            self.__fitness = self._fitness()
        return self.__fitness

    @fitness.setter
    def fitness(self, f):
        self.__fitness = f

    def _fitness(self):
        raise NotImplementedError

    def post_process(self):
        self.__fitness = None

    @classmethod
    def set_fitness(cls, f):
        class C(cls):
            def _fitness(self):
                return f(self)
        return C


class BaseChromosome(BaseFitnessModel):
    default_size = (8,)
    element_class = BaseGene

    def __repr__(self):
        return self.__class__.__name__ + f': {"/".join(repr(gene) for gene in self)}'

    def __str__(self):
        return "/".join(str(gene) for gene in self)

    @classmethod
    def random(cls, size=None):
        raise NotImplementedError


    def cross(self, other):
        raise NotImplementedError

    def merge(self, *other):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def decode(self):
        """Decoding of the chromesome
        Translate the chromesome to (part of) solution, maybe a number.
        """
        return self

    @classmethod
    def encode(cls, x):
        raise NotImplementedError


    def clone(self):
        return self.copy()

    def __eq__(self, other):
        return np.all(self == other)

    def equal(self, other):
        return np.all(self == other)


class BaseIndividual(BaseFitnessModel, metaclass=MetaContainer):
    """base class of individual

    a sequence of chromosomes that may vary in sizes.

    You should implement the methods, cross, mute
    """

    element_class = BaseChromosome
    default_size = 1

    # def __getitem__(self, k):
    #     return self.chromosomes[k]

    def __repr__(self):
        return self.__class__.__name__ + f':= {" $ ".join(repr(chromosome) for chromosome in self.chromosomes)}'

    def __str__(self):
        return self.__class__.__name__ + f':= {" $ ".join(str(chromosome) for chromosome in self.chromosomes)}'

    def __format__(self, spec=None):
        if spec is None:
            return str(self)
        elif spec in {'d', 'decode'}:
            return ' | '.join(str(x) for x in self.decode())
        else:
            return str(self)

    @classmethod
    def random(cls, n_chromosomes=None, *args, **kwargs):
        if isinstance(cls, (MetaList, MetaContainer)):
            if 'sizes' in kwargs:
                return cls([cls.element_class.random(size=size) for size in kwargs['sizes']])
            else:
                if n_chromosomes is None:
                    n_chromosomes = cls.default_size
                return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_chromosomes)])

        elif isinstance(cls, MetaTuple):
            return cls([C.random(*args, **kwargs) for C in zip(cls.element_class)])


    @property
    def chromosomes(self):
        return self.__elements

    @chromosomes.setter
    def chromosomes(self, c):
        self.__elements = c
        self.fitness = None


    def cross(self, other, k=None):
        return self.__class__([chromosome.cross(other_c) for chromosome, other_c in zip(self.chromosomes, other.chromosomes)])

    def mutate(self):
        self.fitness = None
        for chromosome in self.chromosomes:
            chromosome.mutate()

    # def __add__(self, other):
    #     return self.chromosomes +

    def __mul__(self, other):
        return self.mate(other)


    def proliferate(self, k=2):
        inds = [self.clone()] * k
        for i in inds:
            i.mutate()
        return inds


    def clone(self, type_=None, fitness=True):
        if type_ is None:
            type_ = self.__class__
        if fitness is True:
            fitness = self.fitness
        return type_([chromosome.clone() for chromosome in self.chromosomes], fitness=fitness)


    def get_neighbour(self):
        raise NotImplementedError

    def decode(self):
        return [chromosome.decode() for chromosome in self.chromosomes if hasattr(chromosome,'decode')]

    def __getstate__(self):
        return self.chromosomes, self.fitness

    def __setstate__(self, state):
        self.chromosomes, self.fitness = state

    def __eq__(self, other):
        return np.all([c.equal(oc) for c, oc in zip(self.chromosomes, other.chromosomes)])


class BasePopulation(BaseFitnessModel, metaclass=MetaHighContainer):
    """The base class of population in GA
    
    Represents a state of a stachostic process (Markov process)
    
    Extends:
        BaseIterativeModel
    """

    element_class = BaseIndividual
    default_size = 20
        

    _head = 'best individual & fitness & number of individuals'

    params = {'mate_prob':0.7, 'mutate_prob':0.2, 'tournsize':5}

    @property
    def _row(self):
        best = self.best_individual
        return f'{best:d} & {best.fitness} & {self.n_individuals}'

    def __init__(self, individuals=[]):
        self.individuals = [i.clone(type_=self.element_class) if not isinstance(i, self.element_class) else i for i in individuals]
        self.sorted_individuals = []

    def __setstate__(self, state):
        self.element_class = state['element_class']
        self.default_size = state.get('default_size', self.__class__.default_size)
        self.individuals = state['individuals']

    @property
    def individuals(self):
        return self.__elements

    @individuals.setter
    def individuals(self, x):
        self.__elements = x
        self.n_elements = self.n_individuals = len(x)
        self.sorted = False

    @classmethod
    def random(cls, n_individuals=None, *args, **kwargs):
        if n_individuals is None:
            n_individuals = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_individuals)])

    def transitate(self, *args, **kwargs):
        """
        Transitation of the states of population
        Standard flow of the Genetic Algorithm
        """
        self.select()
        self.mate()
        self.mutate()

    def migrate(self, other):
        raise NotImplementedError

    def select(self, n_sel=None, tournsize=None):
        """Select the best individual among `tournsize` randomly chosen
        individuals, `n_sel` times. The list returned contains
        references to the input `individuals`.
        """

        if n_sel is None:
            n_sel = self.default_size
        elif 0 < n_sel <= 1:
            n_sel = int(self.n_individuals * n_sel)
        chosen = []
        rest = self.individuals
        size = tournsize or self.tournsize
        n_rest = self.n_individuals
        for i in range(n_sel):
            if n_rest == 1:
                break
            elif n_rest <= size:
                aspirants = rest
            else:
                aspirants = uniformly_select(rest, size)
            winner = max(aspirants, key=attrgetter('fitness'))
            chosen.append(winner)
            rest.remove(winner)
            n_rest -= 1
        if chosen:
            self.individuals = chosen

    def select_best_individuals(self, n=1):
        # first n best individuals
        if n<1:
            n = int(self.n_individuals * n)
        self.individuals = self.sorted_individuals[-n:]

    def clone(self, type_=None):
        if type_ is None:
            type_ = self.__class__
        return type_([individual.clone(type_=type_.element_class) for individual in self.individuals])

    def parallel(self, func):
        return parallel(func, self.individuals)

    def merge(self, other, select=False, *args, **kwargs):
        self.individuals.extend(other.individuals)
        self.n_individuals += other.n_individuals
        if select:
            self.select(*args, **kwargs)

    def mutate(self, mutate_prob=None):
        for individual in self.individuals:
            if random() < (mutate_prob or self.mutate_prob):
                individual.mutate()

    def mate(self, mate_prob=None):
        offspring = [individual.cross(other) for individual, other in zip(self.individuals[::2], self.individuals[1::2])
        if random() < (mate_prob or self.mate_prob)]
        self.individuals += offspring

    def remove(self, individual):
        self.individuals.remove(individual)
        self.n_individuals -= 1

    def pop(self, k=-1):
        self.individuals.pop(k)
        self.n_individuals -= 1

    def local_search(self, *args, **kwargs):
        for individual in self.individuals:
            individual.evolve(*args, **kwargs)

    def _fitness(self):
        return self.mean_fitness

    @property
    def mean_fitness(self):
        return np.mean([individual.fitness for individual in self.individuals])

    # @property
    # def fitness(self):
    #     return self._fitness()

    @property
    def best_fitness(self):
        return np.max([individual.fitness for individual in self.individuals])

    @property
    def best_individual(self):
        k = np.argmax([individual.fitness for individual in self.individuals])
        return self.individuals[k]

    @property
    def best_(self):
        k = np.argmax([individual.fitness for individual in self.individuals])
        return self.individuals[k]

    def get_best_individuals(self, n=1):
        # first n best individuals
        if n<1:
            n = int(self.n_individuals * n)
        return self.sorted_individuals[-n:]

    @property
    def worst_individual(self):
        k = np.argmin([individual.fitness for individual in self.individuals])
        return self.individuals[k]

    @property
    def worst_(self):
        k = np.argmin([individual.fitness for individual in self.individuals])
        return self.individuals[k]

    @property
    def sorted_individuals(self):
        if self.__sorted_individuals == []:
            ks = np.argsort([individual.fitness for individual in self.individuals])
            self.__sorted_individuals = [self.individuals[k] for k in ks]
        return self.__sorted_individuals

    @sorted_individuals.setter
    def sorted_individuals(self, s):
        self.__sorted_individuals = s

    def sort(self):
        self.individuals = self.sorted_individuals


    def argsort(self):
        return np.argsort([individual.fitness for individual in self.individuals])


    def get_rank(self, individual):
        r = 0
        for ind in self.sorted_individuals:
            if ind.fitness <= individual.fitness:
                r += 1
        r /= self.n_individuals
        individual.ranking = r
        return r

    def ranking(self):
        for k, individual in enumerate(self.sorted_individuals):
            r = 1
            for i in self.sorted_individuals[k+1:]:
                if i.fitness <= individual.fitness:
                    r += 1
            individual.ranking = r / self.n_individuals

    @property
    def solution(self):
        return self.best_

    def cross(self, other):
        k = randint(1, self.n_individuals//2)
        l = randint(1, other.n_individuals//2)
        self.individuals = self.individuals[k:] + other.individuals[:l]
        other.individuals = other.individuals[l:] + self.individuals[:k]

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


class ParallelPopulation(BasePopulation):

    def mutate(self):
        self.parallel('mutate')

    def mate(self, mate_prob):
        offspring = parallel(lambda x: x[0].mate(x[1]), [(a, b) for a, b in zip(self.individuals[::2], self.individuals[1::2])
            if random() < (mate_prob or self.mate_prob)])
        self.individuals += offspring


class BaseSpecies(BaseFitnessModel, metaclass=MetaHighContainer):
    element_class = BasePopulation
    default_size = 2

    params = {'migrate_prob': 0.5}

    @classmethod
    def random(cls, n_populations=None, *args, **kwargs):
        if n_populations is None:
            n_populations = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_populations)])

    def migrate(self, migrate_prob=None):
        for population, other in zip(self.populations[:-1], self.populations[1:]):
            if random() < (migrate_prob or self.migrate_prob):
                population.cross(other)

    @property
    def populations(self):
        return self.__elements

    @populations.setter
    def populations(self, x):
        self.__elements = x
        self.n_elements = self.n_populations = len(x)
        self.sorted = False
        self.fitness = None

    def transitate(self, *args, **kwargs):
        for population in self.populations:
            population.transitate(*args, **kwargs)

    @property
    def best_fitness(self):
        return np.max([np.max([individual.fitness for individual in pop.individuals]) for pop in self.populations])

    @property
    def best_individual(self):
        inds = self.individuals
        k = np.argmax([individual.fitness for individual in inds])
        return inds[k]

    @property
    def best_(self):
        inds = self.individuals
        k = np.argmax([individual.fitness for individual in inds])
        return inds[k]

    @property
    def individuals(self):
        inds = []
        for pop in self.populations:
            inds.extend(pop.individuals)
        return inds


class BaseEnvironment:
    """Base Class of Environment
    main method is evaluate that calculating the fitness of an individual or a population
    """
    def __init__(self, model:BaseIterativeModel):
        self.model = model

    def evaluate(self, x):
        if hasattr(x, 'fitness'):
            return x.fitness
        else:
            raise NotImplementedError

    def exec(self, x, method):
        if hasattr(x, method):
            return getattr(x, method)
        else:
            raise NotImplementedError

    def select(self, pop, n_sel):
        raise NotImplementedError
        

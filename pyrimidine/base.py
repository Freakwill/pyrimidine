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
    select ti, ni from arraies t, n
    sum of ni ~ 10 (for example), while ti are exptected to be not repeated

The opt. problem is
    min sum of {ni} and maximum of frequences in {ti}
    where i are selected indexes.

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
from .utils import choice_uniform, randint
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
    def solution(self):
        raise NotImplementedError('Not define solution for the model!')   

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

    def ezolve(self, n_iter=100, *args, **kwargs):
        # Extreamly eazy evolution method for lazybones
        self.init()
        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.post_process()

    def evolve(self, n_iter=100, per=1, verbose=False, decode=False, stat={'Fitness': 'fitness'}, history=False, *args, **kwargs):
        """Get the history of the whole evolution

        Keyword Arguments:
            n_iter {number} -- number of iterations (default: {100})
            verbose {bool} -- to print the iteration process
            decode {bool} -- decode to the real solution
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            history {bool} -- True for recording history, or a DataFrame object recording previous history.
        
        Returns:
            DataFrame | None
        """

        if verbose:
            if stat:
                _head = 'best solution & {" & ".join(stat.keys())}'
            print('iteration & ' , _head)
            print('-------------------------------------------------------------')
            print(f'0 & {self.solution} & {" & ".join(self._stat(stat).values())}')

        if history:
            import pandas as pd
            history = pd.DataFrame(data={k:[v] for k, v in self._stat(stat).items()})
            flag = True
        elif not isinstance(history, pd.DataFrame):
            raise TypeError('Argument `history` should be a DataFrame object.')
        self.init()
        # n_iter = n_iter or self.n_iter or self.default_n_iter
        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.post_process()
            if flag and (per == 1 or k % per ==0):
                stat_row = self._stat(stat)
                history = history.append(stat_row, ignore_index=True)
            if verbose and (per == 1 or k % per ==0):
                print(f'0 & {self.solution} & {" & ".join(self._stat(stat).values())}')
        return history

    def _stat(self, stat):
        return {k: (getattr(self, s)() if isinstance(getattr(self, s), types.FunctionType) 
                    else getattr(self, s)) if isinstance(s, str) and hasattr(self, s) else s(self) 
                for k, s in stat.items()}

    def get_history(self, n_iter=100, stat=None, history=None, *args, **kwargs):
        """Get the history of the whole evolution

        Keyword Arguments:
            n_iter {number} -- number of iterations (default: {100})
            stat {dict} -- a dict(key: function mapping from the object to a number) of statistics 
                           The value could be a string that should be a method pre-defined.
            history {dict} -- the history of iteration (default: {None})
        
        Returns:
            DataFrame
        """
        print(Warning('This method is deprecated!, use `evolve(history=True, ***)`'))
        if stat is None:
            return
 
        self.init()
        import pandas as pd
        if history is None:
            history = pd.DataFrame(data={k:[(getattr(self, s)() if isinstance(getattr(self, s), types.FunctionType) else getattr(self, s)) if isinstance(s, str) and hasattr(self, s) else s(self)] for k, s in stat.items()})
        elif not isinstance(history, pd.DataFrame):
            raise TypeError('Argument `history` should be a DataFrame object.')

        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.post_process()
            history = history.append({k:(getattr(self, s)() if isinstance(getattr(self, s), types.FunctionType) else getattr(self, s)) if isinstance(s, str) and hasattr(self, s) else s(self) for k, s in stat.items()}, ignore_index=True)
        return history

    def perf(self, n_repeats=10, *args, **kwargs):
        import time
        times = []
        data = None
        for _ in range(n_repeats): 
            cpy = self.clone()
            time1 = time.perf_counter()
            data0 = cpy.evolve(history=True, *args, **kwargs)
            time2 = time.perf_counter()
            times.append(time2 - time1)
            if data is None:
                data = data0
            else:
                data += data0
        return data / n_repeats, np.mean(times)

    def post_process(self):
        pass

    @classmethod
    def config(cls, d):
        cls.params.update(d)
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)

    @classmethod
    def set_size(cls, sz):
        cls.default_size = sz
        return cls

    def clone(self, type_=None, *args, **kwargs):
        raise NotImplementedError
 

# class Solution(object):
#     def __init__(self, value, goal_value=None):
#         self.value = value
#         self.goal_value = goal_value

#     def __str__(self):
#         if self.goal_value is None:
#             return ' | '.join(str(x) for x in self.value)
#         else:
#             return f"{' | '.join(str(x) for x in self.value)} & {self.goal_value}"


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
    """Iterative model with fitness

    The fitness should be stored until the the state of the model is changed.
    
    Extends:
        BaseIterativeModel
    
    Variables:
        __fitness {[type]} -- [description]
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

    def clone(self, type_=None, fitness=None):
        if type_ is None:
            type_ = self.__class__
        if fitness is True:
            fitness = self.fitness
        return type_([i.clone(type_=type_.element_class, fitness=fitness) for i in self], fitness=fitness)

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Fitness':'fitness'}
 
        return super(BaseFitnessModel, self).evolve(stat=stat, *args, **kwargs)


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

    def clone(self, *args, **kwargs):
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


    def __repr__(self):
        # seperate the chromosomes with $ 
        sep = " $ "
        return f'{self.__class__.__name__}:= {sep.join(repr(chromosome) for chromosome in self.chromosomes)}'

    def __str__(self):
        sep = " $ "
        return sep.join(str(chromosome) for chromosome in self.chromosomes)

    def __format__(self, spec=None):
        """ 
        Keyword Arguments:
            spec {str} -- set to be `decode` if need decoding (default: {None})
        
        Returns:
            str
        """
        if spec is None:
            return str(self)
        elif spec in {'d', 'decode'}:
            return ' | '.join(str(x) for x in self.decode())
        else:
            return str(self)

    @classmethod
    def random(cls, n_chromosomes=None, *args, **kwargs):
        """Generate an object of Individual randomly
        
        Arguments:
            **kwargs -- set sizes for the sizes of chromosomes
        
        Keyword Arguments:
            n_chromosomes {number} -- the number of chromosomes (default: {None})
        
        Returns:
            BaseIndividual -- an object of Individual
        """

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
        """Set the fitness to be None, when setting chromosomes of the object
        
        Decorators:
            chromosomes.setter
        
        Arguments:
            c {list} -- a list of chromosomes
        """
        self.__elements = c
        self.fitness = None


    def cross(self, other, k=None):
        # Cross operation of two individual
        return self.__class__([chromosome.cross(other_c) for chromosome, other_c in zip(self.chromosomes, other.chromosomes)])

    def mutate(self):
        # Mutating operation of an individual
        self.fitness = None
        for chromosome in self.chromosomes:
            chromosome.mutate()

    def proliferate(self, k=2):
        # Proliferating operation of an individual
        inds = [self.clone()] * k
        for i in inds:
            i.mutate()
        return inds

    def get_neighbour(self):
        raise NotImplementedError

    def decode(self):
        # To decode an individual
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
    __sorted_individuals = []

    params = {'mate_prob':0.75, 'mutate_prob':0.2, 'tournsize':5}

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Fitness':'fitness', 'Population': self.n_elements}
 
        return super(BaseFitnessModel, self).evolve(stat=stat, *args, **kwargs)


    def __setstate__(self, state):
        self.element_class = state['element_class']
        self.default_size = state.get('default_size', self.__class__.default_size)
        self.individuals = state['individuals']

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
        return len(self)

    @classmethod
    def random(cls, n_individuals=None, *args, **kwargs):
        if n_individuals is None:
            n_individuals = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_individuals)])

    def transit(self, *args, **kwargs):
        """Transitation of the states of population

        It is considered to be the standard flow of the Genetic Algorithm
        """
        self.select()
        self.mate()
        self.mutate()

    def migrate(self, other):
        raise NotImplementedError


    def select_aspirants(self, individuals, size):
        return choice_uniform(individuals, size)

    def select(self, n_sel=None, tournsize=None):
        """Select the best individual among `tournsize` randomly chosen
        individuals, `n_sel` times. The list returned contains
        references to the input `individuals`.
        """

        if n_sel is None:
            n_sel = self.default_size
        elif 0 < n_sel < 1:
            n_sel = int(self.n_individuals * n_sel)
        winners = []
        rest = self.individuals
        size = tournsize or self.tournsize
        n_rest = self.n_individuals
        for i in range(n_sel):
            if n_rest == 0:
                break
            elif n_rest <= size:
                aspirants = rest
            else:
                aspirants = self.select_aspirants(rest, size)
            winner = max(aspirants, key=attrgetter('fitness'))
            winners.append(winner)
            rest.remove(winner)
            n_rest -= 1
        if winners:
            self.individuals = winners

    def select_best_individuals(self, n=1):
        # first n best individuals
        if n<1:
            n = int(self.n_individuals * n)
        self.individuals = self.sorted_individuals[-n:]

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
        self.individuals.extend(offspring)

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
        ks = np.argsort([individual.fitness for individual in self.individuals])
        self.individuals = [self.individuals[k] for k in ks]

    def argsort(self):
        return np.argsort([individual.fitness for individual in self.individuals])

    def get_rank(self, individual):
        r = 0
        for ind in self.sorted_individuals:
            if ind.fitness <= individual.fitness:
                r += 1
            else:
                break
        individual.ranking = r / self.n_individuals
        return individual.ranking

    def rank(self):
        sorted_individuals = [self.individuals[k] for k in self.argsort()]
        k = 0
        while k < self.n_individuals:
            r = 0
            individual = sorted_individuals[k]
            for i in sorted_individuals[k+1:]:
                if i.fitness == individual.fitness:
                    r += 1
                else:
                    break
            for i in sorted_individuals[k:k+r+1]:
                i.ranking = (r + k) / self.n_individuals
            k += r + 1


    @property
    def solution(self):
        return self.best_

    def cross(self, other):
        k = randint(1, self.n_individuals-2)
        self.individuals = self.individuals[k:] + other.individuals[:k]
        other.individuals = other.individuals[k:] + self.individuals[:k]

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


    def dual(self):
        return self.__class__([c.dual() for c in self.chromosomes])


    def evolve(self, stat=None, *args, **kwargs):
        if stat is None:
            stat = {'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
        return super(BaseFitnessModel, self).evolve(stat=stat, *args, **kwargs)



class ParallelPopulation(BasePopulation):

    def mutate(self):
        self.parallel('mutate')

    def mate(self, mate_prob):
        offspring = parallel(lambda x: x[0].mate(x[1]), [(a, b) for a, b in zip(self.individuals[::2], self.individuals[1::2])
            if random() < (mate_prob or self.mate_prob)])
        self.individuals.extend(offspring)


class BaseSpecies(BaseFitnessModel, metaclass=MetaHighContainer):
    element_class = BasePopulation
    default_size = 2

    params = {'migrate_prob': 0.5}

    def _fitness(self):
        return self.mean_fitness

    @property
    def mean_fitness(self):
        return np.mean([_.fitness for _ in self.individuals])

    @classmethod
    def random(cls, n_populations=None, *args, **kwargs):
        if n_populations is None:
            n_populations = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_populations)])

    def migrate(self, migrate_prob=None):
        for population, other in zip(self.populations[:-1], self.populations[1:]):
            if random() < (migrate_prob or self.migrate_prob):
                other.individuals.append(population.best_individual.clone())
                population.individuals.append(other.best_individual.clone())


    @property
    def populations(self):
        return self.__elements

    @populations.setter
    def populations(self, x):
        # Set the fitness to be None, when setting populations of the object
        self.elements = x
        self.sorted = False
        self.fitness = None

    def transit(self, *args, **kwargs):
        for population in self.populations:
            population.transit(*args, **kwargs)
        self.migrate()

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

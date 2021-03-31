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
        return np.dot(n, self.chromosomes[0]), max_repeat(ti for ti, c in zip(t, self.chromosomes[0]) if c==1)

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
from .meta import *


class BaseIterativeModel:

    goal_function = None

    _head = 'best solution & fitness'

    params = {'n_iter': 100}

    
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

    def ezolve(self, n_iter=None, *args, **kwargs):
        # Extreamly eazy evolution method for lazybones
        n_iter = n_iter or self.n_iter
        self.init()
        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.postprocess()

    def evolve(self, n_iter=None, period=1, verbose=False, decode=False, stat={'Fitness': 'fitness'}, history=False, *args, **kwargs):
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

        import pandas as pd

        n_iter = n_iter or self.n_iter
        self.init()

        if verbose:
            if stat:
                _head = f'best solution & {" & ".join(stat.keys())}'
            print('iteration & ' , _head)
            print('-------------------------------------------------------------')
            print(f'0 & {self.solution} & {" & ".join(map(str, self._stat(stat).values()))}')

        if history:
            history = pd.DataFrame(data={k:[v] for k, v in self._stat(stat).items()})
            flag = True
        elif history is False:
            flag = False
        elif not isinstance(history, pd.DataFrame):
            raise TypeError('Argument `history` should be a DataFrame object.')
        # n_iter = n_iter or self.n_iter or self.default_n_iter
        for k in range(1, n_iter+1):
            self.transit(k, *args, **kwargs)
            self.postprocess()
            if flag and (period == 1 or k % period ==0):
                stat_row = self._stat(stat)
                history = history.append(stat_row, ignore_index=True)
            if verbose and (period == 1 or k % period ==0):
                print(f'{k} & {self.solution} & {" & ".join(map(str, self._stat(stat).values()))}')
        return history

    def _stat(self, stat):
        res = {}
        for k, s in stat.items():
            if isinstance(s, str) and hasattr(self, s):
                f = getattr(self, s)
                if isinstance(f, types.FunctionType):
                    res[k] = f()
                else:
                    res[k] = f
            elif isinstance(s, types.FunctionType):
                res[k] = s(self)
            elif isinstance(s, (int, float)):
                res[k] = s
            else:
                raise TypeError(f'The type of stat["{k}"] is not permissible!')
        return res



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
        return type_([i.clone(type_=type_.element_class, fitness=None) for i in self], fitness=fitness)

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Fitness':'fitness'}
 
        return super(BaseFitnessModel, self).evolve(stat=stat, *args, **kwargs)

    def __rshift__(self, t):
        """
        Short for clone method
        """
        return self.clone(type_=t, fitness=True)

    def __rlshift__(self, t):
        return self.clone(type_=t, fitness=True)


class BaseChromosome(BaseFitnessModel):
    default_size = (8,)
    element_class = BaseGene
    gene = element_class

    def __repr__(self):
        return self.__class__.__name__ + f': {"/".join(repr(gene) for gene in self)}'

    def __str__(self):
        return "/".join(str(gene) for gene in self)

    @classmethod
    def random(cls, size=None):
        raise NotImplementedError

    def x(self, other):
        # alias for cross
        return self.cross(other)

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

    # def __eq__(self, other):
    #     return equal(self, other)

    def equal(self, other):
        return np.array_equal(self, other)


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

        if isinstance(cls, (MetaList, MetaContainer)) and not isinstance(cls, MetaTuple):
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

    def x(self, other):
        # alias for cross
        return self.cross(other)

    def cross(self, other, k=None):
        # Cross operation of two individual
        return self.__class__([chromosome.cross(other_c) for chromosome, other_c in zip(self.chromosomes, other.chromosomes)])

    def mutate(self, copy=False):
        # Mutating operation of an individual
        self.fitness = None
        for chromosome in self.chromosomes:
            chromosome.mutate()
        return self

    def proliferate(self, k=2):
        # Proliferating operation of an individual
        inds = [self.clone()] * k
        for i in inds:
            i.mutate()
        return inds

    def get_neighbour(self):
        """to get a neighbour of an individual

        e.g. mutate only one gene
        """
        raise NotImplementedError

    def decode(self):
        """Decode an individual to a real solution

        For example, transform a 0-1 sequence to a real number.
        """
        return [chromosome.decode() for chromosome in self.chromosomes if hasattr(chromosome, 'decode')]

    def dual(self):
        """Get the dual individual
        Applied in dual GA
        """
        raise NotImplementedError

    def __getstate__(self):
        return self.chromosomes, self.fitness

    def __setstate__(self, state):
        self.chromosomes, self.fitness = state

    def __eq__(self, other):
        return np.all([c.equal(oc) for c, oc in zip(self.chromosomes, other.chromosomes)])

    def __mul__(self, n):
        from .population import SGAPopulation
        C = SGAPopulation[self.__class__]
        return C([self.clone() for _ in range(n)])

class BasePopulationModel(BaseFitnessModel):
    """subclass of BaseFitnessModel

    It is consisted of a set of solutions.
    """

    __sorted_individuals = []

    def evolve(self, stat=None, *args, **kwargs):
        """Get the history of the whole evolution
        """
        if stat is None:
            stat = {'Best Fitness':'best_fitness', 'Mean Fitness':'mean_fitness', 'STD Fitness':'std_fitness', 'Population': 'n_elements'}
        return super(BasePopulationModel, self).evolve(stat=stat, *args, **kwargs)

    @property
    def individuals(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def _fitnesses(self):
        return [individual.fitness for individual in self.individuals]


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
        k = np.argmax([getattr(individual, key) for individual in self.individuals])
        return self.individuals[k]

    def get_best_individuals(self, n=1):
        # first n best individuals
        if n < 1:
            n = int(self.n_individuals * n)
        elif not isinstance(n, int):
            n = int(n)
        return self.sorted_individuals[-n:]

    def get_worst(self, key='fitness'):
        k = np.argmin([getattr(individual, key) for individual in self.individuals])
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


class BasePopulation(BasePopulationModel, metaclass=MetaHighContainer):
    """The base class of population in GA
    
    Represents a state of a stachostic process (Markov process)
    
    Extends:
        BasePopulationModel
    """

    element_class = BaseIndividual
    default_size = 20
    hall_of_fame = []

    params = {'mate_prob':0.75, 'mutate_prob':0.2, 'tournsize':5}

    def __str__(self):
        return '\n'.join(map(str, self.individuals))

    # @property
    # def individuals(self):
    #     # alias for attribute `elements`
    #     return self.__elements


    def __getstate__(self):
        return {'element_class':self.element_class, 'default_size':self.default_size, 'individuals':self.individuals, 'params':self.params}


    def __setstate__(self, state):
        self.element_class = state.get('element_class', self.__class__.element_class)
        self.default_size = state.get('default_size', self.__class__.default_size)
        self.individuals = state.get('individuals', [])
        self.params = state.get('params', {})


    @classmethod
    def random(cls, n_individuals=None, *args, **kwargs):
        if n_individuals is None:
            n_individuals = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_individuals)])


    def add_individuals(self, inds:list):
        self.individuals += inds

    def transit(self, *args, **kwargs):
        """Transitation of the states of population

        It is considered to be the standard flow of the Genetic Algorithm
        """
        self.select()
        self.mate()
        self.mutate()

    def migrate(self, other):
        """Migration from one population to another

        Applied in Multi population GA, 
        where the best individual of one populaitonemigrates to another.
        """
        raise NotImplementedError


    def select_aspirants(self, individuals, size):
        # select `size` individuals from the list `individuals` in one tournament.
        return choice_uniform(individuals, size)

    def select(self, n_sel=None, tournsize=None):
        """The standard method of selecting operation in GA
        
        Select the best individual among `tournsize` randomly chosen
        individuals, `n_sel` times.
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

    def parallel(self, func):
        return parallel(func, self.individuals)

    def merge(self, other, n_sel=None):
        """Merge two population.

        Applied in the case when merging the offspring to the original population.
        """
        if isinstance(other, (list, tuple)):
            self.individuals += other
        else:
            self.individuals += other.individuals

        if n_sel:
            self.select(n_sel)

    def mutate(self, mutate_prob=None):
        """Mutate the whole population.

        Just call the method `mutate` of each individual
        
        Keyword Arguments:
            mutate_prob {number} -- the proba. of mutation of one individual (default: {None})
        """
        for individual in self.individuals:
            if random() < (mutate_prob or self.mutate_prob):
                individual.mutate()

    def mate(self, mate_prob=None):
        """Mate the whole population.

        Just call the method `mate` of each individual (customizing anthor individual)
        
        Keyword Arguments:
            mate_prob {number} -- the proba. of mating of two individuals (default: {None})
        """
        
        mate_prob = mate_prob or self.mate_prob
        offspring = [individual.cross(other) for individual, other in zip(self.individuals[::2], self.individuals[1::2])
        if random() < mate_prob]
        self.individuals.extend(offspring)
        self.offspring = self.__class__(offspring)

    def remove(self, individual):
        self.individuals.remove(individual)
        self.n_individuals -= 1

    def pop(self, k=-1):
        self.individuals.pop(k)
        self.n_individuals -= 1

    def local_search(self, *args, **kwargs):
        # call local searching method
        for individual in self.individuals:
            individual.evolve(*args, **kwargs)


    def get_rank(self, individual):
        """get rank of one individual

        Use `rank` if you call it frequently.
        """
        r = 0
        for ind in self.sorted_individuals:
            if ind.fitness <= individual.fitness:
                r += 1
            else:
                break
        individual.ranking = r / self.n_individuals
        return individual.ranking

    def rank(self, tied=False):
        """Rank all individuals
        by fitness increasingly
        """
        sorted_individuals = [self.individuals[k] for k in self.argsort()]
        if tied:
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
        else:
            for k, i in enumerate(sorted_individuals):
                i.ranking = k / self.n_individuals

    def cross(self, other):
        # cross two populations as two individuals
        k = randint(1, self.n_individuals-2)
        self.individuals = self.individuals[k:] + other.individuals[:k]
        other.individuals = other.individuals[k:] + self.individuals[:k]


    def dual(self):
        return self.__class__([c.dual() for c in self.chromosomes])


class ParallelPopulation(BasePopulation):

    def mutate(self):
        self.parallel('mutate')

    def mate(self, mate_prob):
        offspring = parallel(lambda x: x[0].mate(x[1]), [(a, b) for a, b in zip(self.individuals[::2], self.individuals[1::2])
            if random() < (mate_prob or self.mate_prob)])
        self.individuals.extend(offspring)


class BaseSpecies(BasePopulationModel, metaclass=MetaHighContainer):
    element_class = BasePopulation
    default_size = 2

    params = {'migrate_prob': 0.2}

    def init(self):
        for p in self.populations:
            p.init()

    def __str__(self):
        return '\n'.join(map(str, self.individuals))

    def _fitness(self):
        return self.mean_fitness

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

    def get_best_individuals(self, n=1):
        # first n best individuals
        if n < 1:
            n = int(self.n_individuals * n)
        elif not isinstance(n, int):
            n = int(n)
        return self.sorted_individuals[-n:]

    @property
    def individuals(self):
        inds = []
        for pop in self.populations:
            inds.extend(pop.individuals)
        return inds

    def __getstate__(self):
        return {'element_class':self.element_class, 'default_size':self.default_size, 'populations':self.populations, 'params':self.params}


    def __setstate__(self, state):
        self.element_class = state.get('element_class', self.__class__.element_class)
        self.default_size = state.get('default_size', self.__class__.default_size)
        self.populations = state.get('populations', [])
        self.params = state.get('params', {})


class BaseEnvironment:
    """Base Class of Environment
    main method is evaluate that calculating the fitness of an individual or a population
    """
    def __init__(self, model:BaseFitnessModel):
        self.model = model

    def evaluate(self, x):
        if hasattr(x, 'fitness'):
            return x.fitness
        elif hasattr(x, '_fitness'):
            return x._fitness()
        else:
            raise NotImplementedError

    def exec(self, x, method):
        if hasattr(x, method):
            return getattr(x, method)
        else:
            raise NotImplementedError

    def select(self, pop, n_sel):
        raise NotImplementedError

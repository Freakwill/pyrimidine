#!/usr/bin/env python3

"""
This is the core module of pyrimidine. The base classes are defined here.

main classes:
BaseGene: the gene of chromosome
BaseChromosome: sequence of genes, represents part of a solution
BaseIndividual: sequence of chromosomes, represents a solution of a problem
BasePopulation: set of individuals, represents a set of a problem
                also the state of a stachostic process
BaseSpecies: set of population for more complicated optimalization


Subclass the classes and override some main method esp. `_fitness`.

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

from operator import attrgetter
import types
import numpy as np
from toolz import concat
import typing

from .utils import choice_uniform, randint, random
from .errors import *
from .meta import *
from .mixin import *


class BaseGene:

    """Base class of genes
    
    Attributes:
        values (tuple of numbers): the values that a gene takes
    """
    
    values = ()

    def __repr__(self):
        return self.__class__.__name__ + f': {self}'

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(np.random.choice(cls.values, *args, **kwargs))


class BaseChromosome(FitnessMixin, metaclass=MetaArray):
    """Base class of chromosomes

    Chromosome is an array of genes. It is the unit of the GA.

    Attributes:
        default_size (int): the default number of genes in the chromosome
        element_class (TYPE): the type of gene
    """
    
    element_class = BaseGene
    default_size = 8

    alias = {
    "chromosomes": "elements",
    "n_chromosomes": "n_elements"
    }

    def __repr__(self):
        return f'{self.__class__.__name__}: {":".join(map(repr, self))}'

    def __str__(self):
        return ":".join(map(str, self))

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

    def equal(self, other):
        return np.array_equal(self, other)


class BaseIndividual(FitnessMixin, metaclass=MetaContainer):
    """Base class of individuals

    It is essentially a sequence of chromosomes that may vary in sizes.

    You should implement the methods: cross, mute ...
    """

    element_class = BaseChromosome
    default_size = 2
    alias = {"chromosomes": "elements"}

    def __repr__(self):
        # seperate the chromosomes with $ 
        sep = " $ "
        return f'{self.__class__.__name__}:= {sep.join(map(repr, self.chromosomes))}'

    def __str__(self):
        sep = " $ "
        return sep.join(map(str, self.chromosomes))

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
            return ' | '.join(map(str, self.decode()))
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

        if isinstance(cls, MetaTuple):
            return cls([C.random(*args, **kwargs) for C in cls.element_class])
        elif isinstance(cls, MetaList):
            if 'sizes' in kwargs:
                return cls([cls.element_class.random(size=size) for size in kwargs['sizes']])
            else:
                if n_chromosomes is None:
                    n_chromosomes = cls.default_size
                return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_chromosomes)])

    def _fitness(self):
        if hasattr(self, 'environment'):
            return self.environment.evaluate(self)
        else:
            raise NotImplementedError

    def cross(self, other, k=None):
        # Cross operation of two individual
        return self.__class__([chromosome.cross(other_c) for chromosome, other_c in zip(self.chromosomes, other.chromosomes)])

    def mutate(self, copy=False):
        # Mutating operation of an individual
        self.__fitness = None
        for chromosome in self.chromosomes:
            chromosome.mutate()
        return self

    def replicate(self, k=2):
        # Replication operation of an individual
        ind = self.clone()
        ind.mutate()
        return ind

    def get_neighbour(self):
        """to get a neighbour of an individual

        e.g. mutate only one gene
        """
        raise NotImplementedError

    def decode(self):
        """Decode an individual to a real solution

        For example, transform a 0-1 sequence to a real number.
        """
        return [chromosome.decode() for chromosome in self.chromosomes]

    def dual(self):
        """Get the dual individual
        Applied in dual GA
        """
        raise NotImplementedError

    def __getstate__(self):
        return self.chromosomes, self.fitness

    def __setstate__(self, state):
        self.chromosomes, self.__fitness = state

    def __eq__(self, other):
        return np.all([c.equal(oc) for c, oc in zip(self.chromosomes, other.chromosomes)])

    def __mul__(self, n):
        """population = individual * n
        
        Args:
            n (TYPE): positive integer
        
        Returns:
            TYPE: BasePopulation
        """
        assert isinstance(n, np.int_) and n>0, 'n must be a positive integer'
        from .population import StandardPopulation
        C = StandardPopulation[self.__class__]
        return C([self.clone() for _ in range(n)])

    def __add__(self, other):
        return self.__class__([this + that for this, that in zip(self.chromosomes, other.chromosomes)])

    def __sub__(self, other):
        return self.__class__([this - that for this, that in zip(self.chromosomes, other.chromosomes)])

    def __rmul__(self, other):
        return self.__class__([other * this for this in self.chromosomes])


class BasePopulation(PopulationMixin, metaclass=MetaHighContainer):
    """The base class of population in GA
    
    Represents a state of a stachostic process (Markov process)
    
    Extends:
        PopulationMixin
    """

    element_class = BaseIndividual
    default_size = 20
    hall_of_fame = []

    params = {'mate_prob':0.75, 'mutate_prob':0.2, 'tournsize':5}

    alias = {"individuals": "elements",
        "n_individuals": "n_elements",
        "best_individual": "best_element",
        "worst_individual": "worst_element",
        "best_individuals": "best_elements",
        "get_best_individual": "get_best_element",
        "get_best_individuals": "get_best_elements"
    }

    def __str__(self):
        return '&\n'.join(map(str, self))

    @classmethod
    def random(cls, n_individuals=None, *args, **kwargs):
        if n_individuals is None:
            n_individuals = cls.default_size
        return cls([cls.element_class.random(*args, **kwargs) for _ in range(n_individuals)])

    def transition(self, *args, **kwargs):
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
        return choice(individuals, size=size, replace=False)

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
        rest = list(range(self.n_individuals))
        size = tournsize or self.tournsize
        n_rest = self.n_individuals

        for i in range(n_sel):
            if n_rest == 0:
                break
            elif n_rest <= size:
                aspirants = rest
            else:
                aspirants = np.random.choice(rest, size, replace=False)
            _winner = np.argmax([self[k].fitness for k in aspirants])
            winner = aspirants[_winner]
            winners.append(winner)
            rest.remove(winner)
            n_rest -= 1
        if winners:
            self.individuals = [self[k] for k in winners]
        else:
            raise Exception('No winners in the selection!')

    def merge(self, other, n_sel=None):
        """Merge two populations.

        Applied in the case when merging the offspring to the original population.

        `other` should be a population or a list/tuple of individuals
        """

        if isinstance(other, BasePopulation):
            self.individuals += other.individuals
        elif isinstance(other, typing.Iterable):
            self.individuals.extend(other)
        else:
            raise TypeError("`other` should be a population or a list/tuple of individuals")

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

    def local_search(self, *args, **kwargs):
        # call local searching method
        for individual in self.individuals:
            individual.ezolve(*args, **kwargs)

    def get_rank(self, individual):
        """get rank of one individual

        Use `rank` if you call it frequently.
        """
        r = 0
        for ind in self.sorted_:
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
        # sorted_ = [self[k] for k in self.argsort()]
        sorted_ = self.sorted_
        if tied:
            k = 0
            while k < self.n_individuals:
                r = 0
                individual = sorted_[k]
                for i in sorted_[k+1:]:
                    if i.fitness == individual.fitness:
                        r += 1
                    else:
                        break
                for i in sorted_[k:k+r+1]:
                    i.ranking = (r + k) / self.n_individuals
                k += r + 1
        else:
            for k, i in enumerate(sorted_):
                i.ranking = k / self.n_individuals

    def cross(self, other):
        # cross two populations as two individuals
        k = randint(1, self.n_individuals-2)
        self.individuals = self.individuals[k:] + other.individuals[:k]
        other.individuals = other.individuals[k:] + self.individuals[:k]

    def dual(self):
        return self.__class__([c.dual() for c in self])


class BaseMultiPopulation(PopulationMixin, metaclass=MetaHighContainer):
    """Base class of BaseMultiPopulation
    
    Attributes:
        default_size (int): the number of populations
        element_class (TYPE): type of the populations
        elements (TYPE): populations as the elements
        fitness (TYPE): best fitness
        sorted (bool): Description
    """
    
    element_class = BasePopulation
    default_size = 2

    params = {'migrate_prob': 0.2}

    alias = {'positions': 'elements',
    'n_populations': 'n_elements',
    'best_population': 'best_element',
    'worst_population': 'worst_element',
    "get_best_population": "get_best_element",
    "get_best_populations": "get_best_elements"}

    def __str__(self):
        return '\n\n'.join(map(str, self))

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

   def transition(self, *args, **kwargs):
        super().transition(*args, **kwargs)
        self.migrate()

   def best_fitness(self):
        return max(map(attrgetter('best_fitness'), self))

   def get_best_individual(self):
        bests = map(methodcaller('get_best_individual'), self)
        k = np.argmax([b.fitness for b in bests])
        return bests[k]

   @property
    def individuals(self):
        return list(concat(map(attrgetter('individuals'), self)))


class BaseCommunity(BaseMultiPopulation):
    # As an alias of `MultiPopulation`
    def __str__(self):
        return ' @\n\n'.join(map(str, self))


class BaseEnvironment(ContainerMixin, metaclass=MetaContainer):
    element_class = None

    """Base Class of environments

    The main method is `evaluate`, computing the fitness of an individual or a population
    """

    def __init__(self, elements):
        for e in elements:
            e.environment = self

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return evaluate(self, *args, **kwargs)

    def select(self, pop, n_sel):
        raise NotImplementedError

    def __enter__(self, *args, **kwargs):
        globals()['_fitness'] = lambda o: self._evaluate(o.decode())
        globals()['_environment'] = self
        return self

    def __exit__(self, *args, **kwargs):
        if '_fitness' in globals():
            del globals()['_fitness']
        if '_environment' in globals():
            del globals()['_environment']


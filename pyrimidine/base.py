#!/usr/bin/env python3

"""
The base classes are defined here, mainly for implementation of GAs.

This is the core module of pyrimidine. 

Main classes:

    BaseGene: the gene of chromosome
    BaseChromosome: sequence of genes, represents part of a solution
    BaseIndividual: sequence of chromosomes, represents a solution of a problem
    BasePopulation: set of individuals, represents a set of a problem
                    also the state of a stachostic process
    BaseMultiPopulation/BaseCommunity: set of populations for more complicated optimalization
    BaseEnviorenment:

Remark:

    1. Subclass the classes and override some main method esp. `_fitness`.
    2. `BaseGene` is not important, as a wrapper of np.int32 and np.float64
    3. The base classes have been crafted specifically for GA-style algorithms.
      If your novel algorithm differs from GAs, it is advisable to derive from the mixin classes. 

Example:

    select ti, ni from arraies t, n
    sum of ni ~ 10 (for example), while ti are exptected to be not repeated

    The opt. problem is
    min sum of {ni} and maximum of frequences in {ti}
    where i are selected indexes.

    ```
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

    class MyPopulation(StandardPopulation):
        element_class = MyIndividual

    pop = MyPopulation.random(n_individuals=50, size=100)
    pop.evolve()
    print(pop.best_individual)
    ```
"""

from operator import attrgetter
import typing
from random import randint, random

from toolz import concat
from itertools import product
import numpy as np

from .errors import *
from .meta import *
from .mixin import *

from .deco import side_effect


class BaseGene:
    """Base class of genes
    
    Attributes:
        values (tuple of numbers): the values that a gene takes
    """
    
    values = (0, 1)

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

    @property
    def class_name(self):
        if hasattr(self.__class__, '_name'):
            return self.__class__._name
        else:
            return self.__class__.__name__

    def __repr__(self):
        return f'{self.class_name}: {":".join(map(repr, self))}'

    def __str__(self):
        return "|".join(map(str, self))

    def transition(self, *args, **kwargs):
        self.mutate()

    def x(self, other):
        # alias for cross
        return self.cross(other)

    def cross(self, other):
        """crossover operation
        
        Args:
            other (BaseChromosome): another choromosome
        
        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @side_effect
    def mutate(self):
        # mutation operation
        raise NotImplementedError

    def replicate(self):
        # Replication operation of a chromosome
        ind = self.copy()
        ind.mutate()
        return ind

    def decode(self):
        """Decoding of the chromesome

        Translate the chromesome to (part of) solution, maybe a number.
        """
        return self

    def equal_to(self, other):
        """Judge that self == other
        
        Args:
            other (BaseChromosome): another choromosome
        
        Returns:
            bool
        """
        return np.array_equal(self, other)

    @classmethod
    def random(cls, *args, **kwargs):
        raise NotImplementedError


class BaseIndividual(FitnessMixin, metaclass=MetaContainer):
    """Base class of individuals

    It is essentially a sequence of chromosomes that may vary in sizes.

    You should implement the methods: cross, mute ...
    """

    element_class = BaseChromosome
    default_size = 2

    alias = {"chromosomes": "elements",
    "n_chromosomes": "n_elements"}

    def __repr__(self):
        # seperate the chromosomes with $ 
        sep = " $ "
        return f'{self.__class__.__name__}:= {sep.join(map(repr, self))}'

    def __str__(self):
        sep = " $ "
        return sep.join(map(str, self))

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

    def _fitness(self):
        # the concrete method to compute the fitness
        if hasattr(self, 'environment'):
            return self.environment.evaluate(self)
        else:
            raise NotImplementedError

    def transition(self, *args, **kwargs):
        # the defualt `transition` method for individual 
        self.mutate()

    def cross(self, other):
        """Cross operation of two individuals
        
        Args:
            other (BaseIndividual): another individual
        
        Returns:
            BaseIndividual
        """
        return self.__class__([chromosome.cross(other_c) for chromosome, other_c in zip(self, other)])

    def cross2(self, other):
        return self.__class__(list(product(*(chromosome.cross2(other_c) for chromosome, other_c in zip(self, other)))))

    @side_effect
    def mutate(self):
        # Mutating operation of an individual
        for chromosome in self:
            chromosome.mutate()
        return self

    def replicate(self):
        # Replication operation of an individual
        ind = self.copy()
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

        Returns:
            list: a list of codes
        """
        return [chromosome.decode() for chromosome in self]

    def __eq__(self, other):
        return np.all(list(map(np.equal, self, other)))

    def __mul__(self, n):
        """population = individual * n
        
        Args:
            n (int): a positive integer
        
        Returns:
            BasePopulation
        """
        
        assert isinstance(n, np.int32) and n > 0, 'n must be a positive integer'
        C = StandardPopulation[self.__class__]
        return C([self.copy() for _ in range(n)])

    def __add__(self, other):
        """algebraic operation `+` of two individuals
        
        Args:
            other (BaseIndividual): another individual
        
        Returns:
            BaseIndividual: the result indiviudal
        """

        return self.__class__([this + that for this, that in zip(self, other)])

    def __sub__(self, other):
        """algebraic operation `-` of two individuals
        
        Args:
            other (BaseIndividual): another individual
        
        Returns:
            BaseIndividual: the result indiviudal
        """

        return self.__class__([this - that for this, that in zip(self, other)])

    def __rmul__(self, n):
        """algebraic operation `n*` of individual

        It is different with population = individual * n

        Args:
            n (float): the number
        
        Returns:
            BasePopulation
        """

        # assert isinstance(n, np.number)
        return self.__class__([n * this for this in self.chromosomes])


class BasePopulation(PopulationMixin, metaclass=MetaContainer):
    """The base class of population in GA
    
    Represents a state of a stachostic process (Markov process)
    
    Extends:
        PopulationMixin
    """

    element_class = BaseIndividual
    default_size = 20

    params = {'mate_prob':0.75, 'mutate_prob':0.2, 'tourn_size':5}

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

    def transition(self, *args, **kwargs):
        """Transition of the states of population. The core method of the class

        It is considered to be the standard flow of the Genetic Algorithm
        """

        self.select()
        self.mate()
        self.mutate()

    def migrate(self, other):
        """Migration from one population to another

        Applied in Multi population GA, 
        where the best individual of one populaitonemigrates to another.

        Args:
            other (BasePopulation): another population
        
        Returns:
            BasePopulation
        """
        raise NotImplementedError

    def select(self, n_sel=None, tourn_size=None):
        """The standard method of selecting operation in GA
        
        Select the best individual among `tourn_size` randomly chosen
        individuals, `n_sel` times.

        Args:
            n_sel (int|float): the number of individuals that will be selected
            tourn_size (int): the size of the tournament
        """

        if n_sel is None:
            n_sel = self.default_size
        elif 0 < n_sel < 1:
            n_sel = int(self.n_individuals * n_sel)
        if n_sel >= self.n_individuals:
            return
        winners = []
        rest = list(range(self.n_individuals))
        size = tourn_size or self.tourn_size
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

        Args:
            other (BasePopulation): another population
            n_sel (int|float): the number of individuals in the result population
        
        Returns:
            BasePopulation: the result population
        """

        if isinstance(other, BasePopulation):
            self.extend(other.individuals)
        elif isinstance(other, typing.Iterable):
            self.extend(other)
        else:
            raise TypeError("`other` should be a population or a list/tuple of individuals")

        if n_sel:
            self.select(n_sel)

    @side_effect
    def mutate(self, mutate_prob=None):
        """Mutate the whole population.

        Just call the method `mutate` of each individual
        
        Keyword Arguments:
            mutate_prob {float} -- the proba. of mutation of one individual (default: {None})
        """

        for individual in self:
            if random() < (mutate_prob or self.mutate_prob):
                individual.mutate()

    def mate(self, mate_prob=None):
        """To mate the entire population.

        Just call the method `mate` of each individual (customizing anthor individual)
        
        Keyword Arguments:
            mate_prob {float} -- the proba. of mating of two individuals (default: {None})
        """
        
        mate_prob = mate_prob or self.mate_prob
        offspring = [individual.cross(other_individual) for individual, other_individual in zip(self[:-1], self[1:])
        if random() < mate_prob]
        self.extend(offspring)
        return offspring

    def mate_with(self, other, mate_prob=None):
        """To mate with another population.

        Just call the method `mate` of each individual (customizing anthor individual)
        
        Keyword Arguments:
            other {BasePopulation} -- another population
            mate_prob {float} -- the proba. of mating of two individuals (default: {None})

        Returns:
            BasePopulation: the offspring
        """
        mate_prob = mate_prob or self.mate_prob
        offspring = [individual.cross(other_individual) for individual, other_individual in product(self, other)
        if random() < mate_prob]
        self.extend(offspring)
        return offspring

    @side_effect
    def local_search(self, max_iter=2):
        """Call local searching method
        
        By default, it calls the `ezolve` methods of individuals, iteratively
        """

        for individual in self:
            individual.ezolve(max_iter=self.get('local_iter', max_iter), initialize=False)

    def get_rank(self, individual):
        """Get rank of one individual

        Use `rank` if you call it frequently.

        Args:
            individual (BaseIndividual, optional): an individual in the population

        Returns:
            float: the rank of the individual
        """

        r = 0
        for ind in self.sorted_():
            if ind.fitness <= individual.fitness:
                r += 1
            else:
                break
        individual.ranking = r / self.n_individuals
        return individual.ranking

    def rank(self, tied=False):
        """To rank all individuals by the fitness increasingly
        
        Args:
            tied (bool, optional): for tied ranking
        """
        
        sorted_list = self.sorted_()
        if tied:
            k = 0
            while k < self.n_individuals:
                r = 0
                individual = sorted_list[k]
                for i in sorted_list[k+1:]:
                    if i.fitness == individual.fitness:
                        r += 1
                    else:
                        break
                for i in sorted_list[k:k+r+1]:
                    i.ranking = (r + k) / self.n_individuals
                k += r + 1
        else:
            for k, i in enumerate(sorted_list):
                i.ranking = k / self.n_individuals

    def cross(self, other):
        """Cross two populations as two individuals

        Args:
            other (BasePopulation): another population
            n_sel (int|float): the number of individuals in the result population
        
        Returns:
            BasePopulation: the result population
        """

        k = randint(1, self.n_individuals-2)
        return self.__class__(self[k:] + other[:k])

    def migrate(self, other):
        """Migrate between two populations
        
        Args:
            other (BasePopulation): another population
            n_sel (int|float): the number of individuals in the result population
        """

        k = randint(1, self.n_individuals-2)
        self.individuals = self[k:] + other[:k]
        other.individuals = other[k:] + self[:k]


class BaseMultiPopulation(MultiPopulationMixin, metaclass=MetaHighContainer):
    """Base class of BaseMultiPopulation
    
    Attributes:
        default_size (int): the number of populations
        element_class (BasePopulation): the type of the populations
    """
    
    element_class = BasePopulation
    default_size = 2

    params = {'migrate_prob': 0.75}

    alias = {'populations': 'elements',
    'n_populations': 'n_elements',
    'get_best_populationspopulation': 'best_element',
    'worst_population': 'worst_element',
    'get_best_population': 'get_best_element',
    'get_best_populations': 'get_best_elements'
    }

    def __str__(self):
        return '\n\n'.join(map(str, self))

    def migrate(self, migrate_prob=None, copy=True):
        migrate_prob = migrate_prob or self.migrate_prob
        for population, other in zip(self[:-1], self[1:]):
            if random() < migrate_prob:
                other.append(population.get_best_individual(copy=copy))
                population.append(other.get_best_individual(copy=copy))

    def transition(self, *args, **kwargs):
        self.migrate()
        for p in self:
            p.transition(*args, **kwargs)

    @side_effect
    def select(self):
        # select the multi-population
        for p in self:
            p.select()

    @side_effect
    def mutate(self):
        # mutate the multi-population
        for p in self:
            p.mutate()

    def mate(self):
        raise NotImplementedError

    def get_best_individual(self, copy=True):
        """To get the individual with the max. fitness
        
        Args:
            copy (bool, optional): if it is true, then return a copy

        Return:
            BaseIndividual: An individual representing the solution
        """

        bests = map(methodcaller('get_best_individual'), self)
        k = max(b.fitness for b in bests)
        if copy:
            return bests[k].copy()
        else:
            return bests[k]

    @property
    def individuals(self):
        return list(concat(map(attrgetter('individuals'), self)))


class BaseCommunity(BaseMultiPopulation):
    # An alias of `MultiPopulation`

    def __str__(self):
        return ' @\n\n'.join(map(str, self))


class BaseEnvironment(CollectiveMixin, metaclass=MetaContainer):

    """Base Class of environments

    The main method is `evaluate`, computing the fitness of an individual or a population
    """

    element_class = None

    def __init__(self, elements=[]):
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


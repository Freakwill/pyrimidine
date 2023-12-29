# API Design

## Modules

Short introduction to modules:

- `meta`: Metaclasses
- `mixin`: Mixin Classes
- `base`: The base classes, esp. for creating GAs
- `population`: populations to implement classical GAs
- `chromosome`: some type of chromosomes
- `multipopulation`: multi-population GAs
- `individual`: individuals for GA or other EA
- `es/de/ep`: Evolutionary algorithms
- `pso/ba/fa`: Swarm intelligent algorithms
- `deco`: Decorators
- `utils`: Helpers
- `errors`: Exceptions
- `local_search/*`: Local search algorithms
- `learn/*`: Machine learning implemented by GAs

## Metaclasses
The metaclass `System` is defined to simulate abstract algebraic systems, which are instantiated as a set containing a set of elements, as well as operators and functions on them.

`MetaContainer` is the super-metaclass of `System` for creating container classes.

1. `ParamType --> MetaContainer --> System`
2. `ParamType --> MetaArray`

`MetaArray` is a compact version of `MetaContainer`

## Mixin Classes

Metaclasses define what the algorithm is, while mixin classes specify what the algorithm does. 

The inheritance of metaclasses:

```
IterativeMixin  - - ->  CollectiveMixin
    |                      |
    |                      |
    v                      v
FitnessMixin  - - ->  PopulationMixin
```

If you want to create novel algorithm different from GAs, it is recommended to inherit from the mixin classes. The base classes are designed for GA-like algorithms. It is not coercive. One can override the methods in the base classes.

### Iterative models/algorithms

An iterative model is a mixin class implementing iterative algorithms, mainly repeating to call `transition` method.

Following is the core (pseudo-)code of the class.

```python
class IterativeMixin:

    def evolve(self):
        self.init()
        looping by k:
            self.transition(k)
```

Users could override `transition` to implement own iteration algorithms.


### Fitness models

Fitness Models (`FitnessMixin`) are iterative models with fitness. The iteration in such models would be influenced by fitness.

Indiviudal is a fitness model in GA.

### Population models

Population Models (`PopulationMixin`) are collective algorithms with fitness.


## Basic Classes

### Individuals

An individual is defined as a container of chromosomes.

The individual classes are inherited from `BaseIndividual`. For instance, `BinaryIndividual` is a subclass encoded by   several binary chromosomes.


See `Examples` section for a simple example of binary individuals --- Knapsack Problem.


### Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.  The subclass used most frequently is`BinaryChromosome`.

A chromosome is equivalent to an individual with only one chromosome, mathematically.


### Populations

The population classes are inherited from `BasePopulation`. `StandardPopulation` is the standard population for GA. It is recommended to use `HOFPopulation` in most cases.

A population is a container of individuals, in original meaning. But It is possible to be a container of chromosomes in the view of algebra.


### Multi-populations

It is useful for multi-populations GA. It is regarded as a container of populations.


### Environment
It is designed to be the context for evolution of individuals. The aim of the class is not for numerical computing, instead for "the Skeuomorphic Design".

## Methods

### About fitness
Calculating the fintess is the most frequent task in the algorithms.

- `_fitness()`: the basic method to calculate the fitness of the individual. The methods whose names are started with `_` always do the most dirty task.
- `fitness`: decorated by `property`, get the fitness of the individual/population (from the cache first then call - `_fitness` if there is a cache). The fitness of a population is the maximal fitness of the individuals in the population.
- `fitness`: decorated by `property`, get the fitness of the individual (from the cache first then call `_fitness` if there is a cache).
- `best/mean/std_fitness`: the maximum/mean/std of the fitnesses of the individuals in a population.
- `get_all_fitness`: get the all fitnesses

### About elements(individuals in a population or populations in a multi-population/community)

- `best_element`: wrapped by `property`, equiv. to `get_best_element`
- `get_best_element`: get the element with maximal fitness
- `get_best_elements`: get some elements
- `sort/argsort`: sort the elements by fitness
- `solution`: decoding the best element

### Basic operation

- `copy`: copy an object
- `random`: class method to generate an object randomly
- `decode`: decode an individual to the real solution

### Genetic operation
- `clone`: chromosome-level
- `mutate`: chromosome-level
- `cross`: chromosome-level
- `select`: population-level
- `mate`: population-level, cross the elements, invalid in solution-level.
- `migrate`: population-level, crossover of the populations
- `dual`: chromosome-level
- `replicate`: copy then mutate
- `local_search`: population-level

Some operators defined in solution-level, i.e. only valid for individuals/populations (or chromosomes as the individuals), such as `_fitness`, some could be defined under solution-level, i.e. valid for chromosomes in an individual, such as `dual`, `cross`, `mutate`.

### List-like operation
`append, extend, pop, remove` to operate the list of the elements

### Side-effect and pure

Here `side-effect` means the method will change the fitness of the individual or the population. It could be ignored if you do not use the cache (to get the fitness).

### Others

- `set_*`/`get_*`: set/get methods, for instance, `get_all` the attributes of all individuals in a population.
- `set`: set attributes/methods for a class
- `save/load`: serialization/deserialization for populations/individuals

## Arguments/Parameters/Attributions

- `n_*`: number of ..., such as `n_chromosomes` represents the number of the chromosomes in an individual.
- `*_prob`: probability
- `*_rate`: e.g. `learning_rate`

## `params`/`alias`

`params` is a dictionary of parameters for the algorithm, it could be inherited from super class by meta-programming.

`alias` is a dictionary for alias to attributes.
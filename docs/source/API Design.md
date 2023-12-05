# API Design

## Iterative Models

An iterative model is a class implementing iterative algorithms, repeating to call $x'=Tx$.

Following is the core code of the class.

```python
def evolve(self):
    self.init()
    looping:
        self.transition()
```

Users could override `transition` to implement any other iteration form.


### Fitness Models

Fitness Models are Iterative Models with fitness. Iteration in fitness models would be influenced by fitness.

Indiviudal is a fitness model in GA.

## Individuals

An individual is defined as a container of chromosomes.

The individual classes are inherited from `BaseIndividual`. For instance, `BinaryIndividual` is a subclass encoded by   several binary chromosomes.



See `Examples` section for a simple example of binary individuals --- Knapsack Problem.


## Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.  The subclass used most frequently is`BinaryChromosome`.

A chromosome is equivalent to an individual with only one chromosome, mathematically.


## Populations

The population classes are inherited from `BasePopulation`. `StandardPopulation` is the standard population for GA. It is recommended to use `HOFPopulation` in most cases.

A population is a container of individuals, in original meaning. But It is possible to be a container of chromosomes in the view of algebra.


## Multi-populations

It is useful for multi-populations GA. It is regarded as a container of populations.


## Environment
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

### Basic Operation

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

### Others

`set_*`/`get_*`: set/get methods, for instance, `get_all` the attributes of all individuals in a population.
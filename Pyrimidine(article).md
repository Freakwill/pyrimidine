Journal: SOFTWARE DEVELOPMENT & APPLICATION
date: 2021.1
---

# `Pyrimidine`: Object-Oriented Genetic Algorithm Framework in Python


**Abstract**: `Pyrimidine` is a versatile framework developed for implementing genetic algorithms. It can also implement any iterative model, such as simulated annealing or particle swarm optimization. Its design is based on object-oriented programming, leveraging the metaprogramming capabilities of Python to treat populations as containers of individuals, individuals as containers of chromosomes, and containers as metaclasses for constructing different structures of individual and population classes. Due to its highly object-oriented nature, it is easily extensible and adaptable.

**Keywords**: `Pyrimidine` framework; Genetic algorithm; Python language; Object-oriented programming; Metaprogramming

## Introduction

Genetic algorithms are a universal optimization method that mimics natural selection to solve optimization problems. They are among the earliest developed intelligent algorithms [1-3], widely used in various fields, and have been modified and combined with new algorithms [4]. This paper does not delve into the principles of genetic algorithms; for more details, refer to the literature [5,6] and references therein.

Currently, several libraries in different languages provide frameworks for implementing genetic algorithms. Python, in particular, offers multiple genetic algorithm libraries, including the well-known DEAP [7], gaft, tpot [8,9] for machine learning parameter optimization, and scikit-opt and gplearn as extensions of scikit-learn [10]. This paper introduces the design of the general algorithm framework `Pyrimidine`. It strictly adheres to the object-oriented programming paradigm and utilizes the metaprogramming features of Python more rigorously than other libraries.

## Framework Design

The framework design primarily consists of two parts: classes implementing the fundamental concepts of genetic algorithms, such as individuals and populations, and metaclasses used to construct these classes.

In the concrete implementation using Python, containers are object lists (or other feasible iterators). For instance, a population is a list of individuals; an individual is a list of chromosomes; and a chromosome is an array of genes. The specific implementation of the gene array is crucial and can be done using the standard library `array` or the well-known third-party numerical computing library numpy. The latter is more convenient for various numerical computations but has slower crossover operations.

## Container Metaclass

This metaclass is inspired by functional programming languages like Haskell and type theory. There are two main types of containers: lists and tuples, where lists represent elements with the same type, and tuples have no such restrictions. Users are free to modify the metaclass, so detailed content is not extensively covered here.

Among them, `BaseIndividual` is the base class for all individual classes. This notation is influenced by variable types, such as a list of strings written as `List[String]`.

## Basic Classes

In `pyrimidine`, there are three fundamental classes: `BaseChromosome`, `BaseIndividual`, and `BasePopulation`, representing chromosomes, individuals, and populations, respectively. As mentioned earlier, individuals can have multiple chromosomes, which differs from the typical design of genetic algorithms. The general sequence when designing an algorithm is to inherit from `BaseChromosome`, construct the user's chromosome class, then inherit from `BaseIndividual`, construct the user's individual class, and finally use the same method to construct the population class. For user convenience, `pyrimidine` provides some commonly used subclasses, eliminating the need for users to repeatedly set up. By inheriting from these classes, users automatically gain encoding schemes for solutions, crossover and mutation methods for chromosomes. Genetic algorithms typically use binary encoding, and `pyrimidine` provides `BinaryChromosome` for this purpose.

By inheriting from this class, users obtain binary encoding, as well as algorithms for two-point crossover and independent mutation for each gene.

Typically, users start designing algorithms with the following subclasses. Among them, `MonoIndividual` enforces that an individual can only have one chromosome (although not mandatory for the algorithm, and `BaseIndividual` can be used instead).

```python
class BaseChromosome:
    # Chromosome class definition

class BaseIndividual:
    # Individual class definition

class BasePopulation:
    # Population class definition

class MonoIndividual(BaseIndividual):
    # Individual class with a single chromosome
```

These classes serve as the foundation for constructing genetic algorithms in `pyrimidine`. Users can then extend and customize them based on the specific requirements of their optimization problems.

```python
class MyIndividual(MonoIndividual): 
    element_class = BinaryChromosome 
    def _fitness(self):
        # compute the fitness
```

`MyIndividual` is a base class with `BinaryChromosome` as its chromosome. The subsequent base classes are all constructed by metaclasses. Utilizing methods provided by metaclasses, users can construct an individual class composed of several `ExampleChromosome` chromosomes. As `MonoBinaryIndividual` is such a class, an equivalent expression is:

```python
class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        # Fitness computation
```

Using a standard genetic algorithm as the iterative method, we directly set `MyPopulation = SGAPopulation[MyIndividual]`. This notation, inspired by Python's typing module (and the type concept in Haskell), is implemented through metaprogramming, signifying that MyPopulation is a list composed of `MyIndividual` objects.

If there's a need to redesign crossover and mutation algorithms, you can override the `mutate` and `cross` methods in the individual class. It's also possible to override methods in the chromosome class since the genetic operation methods of individual class objects simply invoke the methods of each chromosome class object.

These classes inherit from the `BaseIterativeModel` base class, which standardizes the iteration format, including exporting data for visualization. Developing new algorithms involves overriding the `transit` method, as all iterative algorithms repeatedly call this method. For genetic algorithms, `transit` primarily involves executing various genetic operations successively.

## Example

Illustrating the basic usage of `Pyrimidine` with a simple example: the classical 50-dimensional knapsack problem:

$$ \max \sum_{i=1}^{n} C_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} W_i x_i \leq W 
$$

This problem can be directly encoded in binary without requiring decoding, making it suitable for testing genetic algorithms. The problem is defined in the sub-module `pyrimidine.benchmarks.optimization` under `Knapsack.random`. This function generates suitable parameters $C_i$, $W_i$, and $W$. Users can replace it with their own optimization problems by overriding the `_fitness` method.

### Algorithm Construction

Using the classes provided by `pyrimidine`, it is straightforward to construct a population with 20 individuals, each containing a 50-dimensional chromosome. The population will iterate 100 times, and the fittest individual in the last generation will be the solution to the optimization problem.

```python
from pyrimidine import MonoBinaryIndividual, SGAPopulation
from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)  # Mapping n-dimensional binary encoding to the objective function value

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        # Return value must be a number
        return _evaluate(self.chromosome)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 20

pop = MyPopulation.random(size=n)  # Size: length of the chromosome
pop.evolve(max_iter=100)
```

Finally, the optimal individual can be found using `pop.best_individual` as the solution. Setting `verbose=True` prints the iteration process. The equivalent expression is as follows:

```python
MyPopulation = SGAPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))]
pop = MyPopulation.random(n_individuals=20, size=n)
pop.evolve()
```

The equivalent expression no longer explicitly depends on class inheritance and the class syntax, making the program more "algebraic" and concise.

### Visualization

To assess the algorithm's performance, it is common to plot fitness curves or other metric sequences over iterations, which can be done using the `history` method. This method returns a `pandas.DataFrame` object that records statistical results for each generation of the population. Users can use it to freely plot performance curves. Generally, users need to provide a "statistics dictionary" where the keys are the names of the statistics, and the values are functions (limited to predefined population methods or attributes with numeric return values). See the code snippet below:

```python
stat = {'Mean Fitness': 'mean_fitness', 'Best Fitness': 'best_fitness'}
data = pop.history(stat=stat, max_iter=100)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```

Here, the strings `mean_fitness` and `best_fitness` respectively represent the population's average fitness and the fittest individual's fitness. Of course, they are ultimately implemented by predefined object methods, such as `best_fitness`, which maps `pop` to `pop.best_individual.fitness`.

### Algorithm Extension

Developing new algorithms with `pyrimidine` is straightforward. In classical genetic algorithms, the mutation rate and crossover rate are consistent for the entire population, but `pyrimidine` can easily encode them into each individual, allowing them to change over iterations.

```python
class NewIndividual(MixIndividual):
    element_class = (BinaryChromosome, FloatChromosome)

    def mutate(self):
        # Mutate based on the first bit of the second chromosome self[1][0]
    
    def cross(self, other):
        # Crossover based on the second bit of the second chromosome self[1][1]
    
    def _fitness(self):
        # Fitness is only related to the first chromosome
        f(self[0])

class NewPopulation(SGAPopulation):
    element_class = NewIndividual
    default_size = 20

# 8 represents the encoding length of variables, and 2 represents mutation and crossover rates
pop = NewPopulation.random(sizes=(8, 2))
```

`FloatChromosome` comes pre-equipped with genetic operations for floating-point numbers, eliminating the need for user-defined specifications. This way, a genetic algorithm with evolving mutation and crossover rates can be developed.

## Conclusion

Extensive experiments and improvements demonstrate that `pyrimidine` is a versatile framework suitable for implementing various genetic algorithms. Its design features strong extensibility, allowing the implementation of any iterative model, such as simulated annealing or particle swarm optimization. For users developing new algorithms, `pyrimidine` is a promising choice.

Currently, `pyrimidine` is still in development, but most APIs have been finalized, alleviating concerns about changes. `Pyrimidine` requires fitness values to be numbers, so it cannot handle multi-objective problems directly, unless they could be reduced to single-objective problems. However, the implementation of multi-objective optimization is under consideration. `Pyrimidine` uses numpy's numerical classes, making crossover operations slower than DEAP's, but alternative implementations can be employed. Of course, there are other areas for improvement. Additionally, `pyrimidine`'s documentation is still in progress.

## References


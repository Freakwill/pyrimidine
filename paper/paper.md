---
title: 'Pyrimidine: Algebra-inspired Programming framework for evolution algorithms
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Congwei Song
    orcid: 0000-0002-4409-7276
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib
---

# Pyrimidine: Algebra-inspired Programming framework for genetic algorithms

**Abstract** Pyrimidine is a general framework for genetic algorithms. It is extremely extensible and can implement any iterative model, such as simulated annealing and particle swarm optimization. Its design is based on object-oriented programming and fully utilizes the metaprogramming capabilities of Python. We propose a container metaclass to construct different structures of individual and population classes. These classes are understood as algebraic systems, where elements can perform various operations, such as mutation and crossover of individuals in a population. These classes may also be the elements of higher-order classes, allowing for automatic implementation of class-level operations such as population migration in genetic algorithms. We call such design style "the algebra-inspired Progamming".


**Keywords** Pyrimidine, Genetic Algorithms, Algebra-inspired Programming, Python, meta-programming

## Introduction

As one of the earliest developed intelligent algorithms [1-4], GA has found extensive application across various domains and has undergone modifications and integrations with new algorithms [5-6]. The principles of GA will not be extensively reviewed in this article. For a detailed understanding, please refer to reference [4] and the associated literatures.

Presently, a variety of programming languages feature libraries that implement GA frameworks. Python stands out for its extensive collection of GA frameworks, including notable ones like deap [7] for general purposes, gaft for optimization, and tpot for super-parameter tuning [8-9], along with scikit-learn, such as scikit-opt and gplearn [10]. 

This article introduces `pyrimidine`, a general algorithm framework for GA and any other evoluationary algorithm. Adhering rigorously to object-oriented programming (OOP) principles, `pyrimidine` distinguishes itself from other libraries, making effective use of Python's metaprogramming capabilities.

## Algebra-inspired Programming

As known, GA consists of two main components: individuals( or chromosomes) and populations.

In a typical Python implementation, populations are initially conceptualized as lists of individuals, with each individual representing a chromosome composed of a list of genes. Subsequently, creating an individual can be achieved using either the standard library's `array` or the widely-used third-party library `numpy`[11]. Finally, the evolutionary operators are defined and applied to these structures.

Our design concept is beyond the oridinary idea and more extensible. We would like to call it "algebra-inspired Programming". It should not be confused with algebraic programming, but we can draw inspiration from its ideas.

### Mathematical representation

We introduce the concept of a **container**, simulating an abstract algebraic system where specific operators are not yet defined.

We represent a **container** $s$ of type $S$, with elements of type $A$, using the following expression:
$$
s = \{a:A\}:S
$$
In this context, $\{\cdot\}$ denotes either a set or a sequence (emphasizing the order of the elements).

Building upon this concept, we define a population as a container of individuals. And it is straightforward to introduce the notion of a multi-population as a container of populations, referred to as the high-level container.

A container that defines operators of its elements is termed a **system**. For example, we define the operation `S.cross(a, b)` on the elements $a,b$ in the system $S$ to implement the crossover operation of two individuals in a population. It shares similarities with algebraic systems. However, the current version does not incorporate this concept, operations are directly defined as methods of the elements, such as `a.cross(b)`. The contemplation of incorporating this concept is deferred to future versions, and this prospective change will not impact the design of APIs.

An individual may be viewed as a container of chromosomes, but it will not to be a system. A chromosome can be perceived as a container of genes, while in practice, we implement chromosomes directly using `numpy.array` or the standard library's `array.array`.

The lifting of a function/method $f$ is defined as:
$$
f(s) := \{f(a)\}
$$
unless explicitly redefined. For instance, the mutation of a population entails the mutation of all individuals in it, but at times, it may be defined as the mutation of one individual selected randomly.

Some methods may be lifted differently, such as:
$$
f(s) := \max_t\{f(t)\}
$$
A notable example is `fitness`, used to compute the fitness of the entire population.

`transition` is the primary method in the iterative algorithms, denoted as a transform:
$$
T(s):S\to S
$$
The iterative algorithms can be represented as $T^n(s)$.
And if $s$ is a container, then $T(s)=\{T(a)\}$ by default where $T(a)$ is pre-defined.

### Metaclasses
We define the metaclass `System` to simulate abstract algebraic systems, which are instantiated as a set containing a set of elements, as well as operators and functions on them.

`Container` is a super-metaclass of `System` for creating containers.

There are mainly two types of containers: list-like and tuple-like, where the former implies that all elements in the container are of the same type, while the later has no such restriction.

### Fundamental Classes

There are three fundamental classes in `pyrimidine` constructed by the metaclasses: `BaseChromosome`, `BaseIndividual`, `BasePopulation`, to create chromosomes, individuals and populations respectively.

Constructing an individual, `SomeIndividual`, consisting of several `SomeChromosome`, is as simple as defining `SomeIndividual = BaseIndividual[SomeChromosome]`, where `BaseIndividual` is the base class for all individual classes. Similarly, a population of `SomeIndividual` could be `BasePopulation[SomeIndividual]`.

For convenience, `pyrimidine` provides some commonly used subclasses, so users do not have to redefine these settings. By inheriting these classes, users gain access to the methods such as, cross and mutation. Genetic algorithms generally use binary encoding. `pyrimidine` offers `BinaryChromosome` for the binary settings. By inheriting from this class, users have binary encoding and algorithm components for two-point cross and independent gene mutation.

Generally, users start the algorithm design as follows, where `MonoIndividual` enforces that individuals can only have one chromosome.

```python
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome # default size of the chromosome is 8
    def _fitness(self):
        # Compute the fitness

class MyPopulation(StandardPopulation):
    element_class = MyIndividual
```

`MyIndividual` is a class (an concrete container) with `BinaryChromosome` as its chromosome type. As you saw, it is equivalent to `MyIndividual=MonoIndividual[BinaryChromosome]`. Since `binaryIndividual(size=8)` creates such a class, an equivalent way to write this is,

```python
class MyIndividual(binaryIndividual()):
    def _fitness(self):
        # Compute the fitness
```

By using a standard GA as the iteration method, you can directly set `MyPopulation = StandardPopulation[MyIndividual]`. It's implemented through metaprogramming and signifies that `MyPopulation` is a list-type container of `MyIndividual` objects.

Algebraically, there is no different between `MonoIndividual` and a single `Chromosome`. And the population also can be a container of chromosomes.

```python
class MyChromosome(BaseChromosome):
    def _fitness(self):
        # Compute the fitness

class MyPopulation(StandardPopulation):
    element_class = MyChromosome
```



### Mixin classes

These classes also inherit from the mixin class `IterativeMixin`, responsible for the iteration, including data export for visualization. When developing a new algorithm, the crucial step is to override the `transition` method, which is invoked in all iterative algorithms. In the context of genetic algorithms, the `transition` method primarily comprises `mutate` and `cross`, representing the crossover and mutation methods.

As subclasses of `IterativeMixin`, `FitnessMixin` is created to execute the iterative algorithm aiming to maximize fitness, while `ContainerMixin` and `PopulationMixin` represent their "swarm" forms.

Metaclasses define what the algorithm is, while mixin classes specify what the algorithm does. When designing a new algorithm that may differ from GA, it is recommended to inherit from the mixin classes initially.

Four mixin classes are presented below, along with the corresponding inheritance arrows.

```
IterativeMixin  --->  ContainerMixin
    |                      |
    |                      |
    v                      v
FitnessMixin  --->  PopulationMixin
```


## An Example to start

In this section, we demonstrate the basic usage of `pyrimidine` with a simple example: the classic 0-1 knapsack problem with $n=50$ dimensions. (See more examples on GitHub)

$$
\max \sum_i c_ix_i \\
\sum_i w_ix_i \leq W, \quad x_i=0,1
$$

The problem solution can be naturally encoded in binary format without requiring additional decoding.


```python
from pyrimidine import MonoIndividual, StandardPopulation, BinaryChromosome
from pyrimidine.benchmarks.optimization import Knapsack
# The problem is defined in the submodule `pyrimidine.benchmarks.optimization`, so we import it directly, but user can redefine it manually.

n = 50
_evaluate = Knapsack.random(n)  # Function mapping n-dimensional binary encoding to the objective function value

class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome
    default_size = n
    def _fitness(self):
        # The return value must be a number
        return _evaluate(self.chromosome)

"""
equivalent to:
MyIndividual = MonoIndividual[BinaryChromosome].set_fitness(lambda o: _evaluate(o.chromosome)) // n

or with the helper `binaryIndividual`: MyIndividual = binaryIndividual(size=n).set_fitness(lambda o: _evaluate(o.chromosome))
"""

class MyPopulation(StandardPopulation):
    element_class = MyIndividual
    default_size = 20

"""
equivalent to:
MyPopulation = StandardPopulation[MyIndividual] // 20
"""

pop = MyPopulation.random()
pop.evolve(n_iter=100) # Setting `verbose=True` will print the iteration process.
```

Finally, the optimal individual can be obtained with `pop.best_individual` or `pop.solution`, as the solution of the problem.

The equivalent expressions no longer explicitly depends on class inheritance, making the code more concise and similar to algebraic style.

## Visualization

Instead of implementing visualization methods, `pyrimidine` yields a `pandas.DataFrame` object that encapsulates statistical results for each generation by setting `history=True` in `evolve` method. Users can harness this object to create customizable performance curves. Generally, users are required to furnish a "statistic dictionary" whose keys are the names of the statistics, and values are functions mapping the population to numerical values (strings are confined to pre-defined methods or attributes of the population).

```python
# statistic dictionary, computing the mean fitness and best fitness of each generation
stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}

# obtain the history data, i.e. the statistical results, through the evolution.
data = pop.evolve(stat=stat, n_iter=100, history=True)

# draw the results
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```

Here, `mean_fitness` and `best_fitness` denote the average fitness value of the population and the optimal individual fitness value, respectively. Notably, they inherently encapsulate functions to perform statistical operations, for instance, `best_fitness` corresponds to the mapping `pop->pop.best_individual.fitness`.

![](/Users/william/Programming/myGithub/pyrimidine/plot-history.png)


## Create your own classes and algorithms
In standard GAs, the mutation rate and crossover rate remain constant and uniform throughout the entire population during evolution. However, in self-adaptive GAs, these rates can be dynamically encoded in each individual, allowing for adaptability during iterations. It is remarkably simple to implement self-adaptability by `pyrimidine`. 

We introduce an "mixed-individual" comprising two chromosomes of different types: one representing the solution and the other encapsulating the probabilities of mutation and crossover.

```python
class NewIndividual(MixedIndividual):
    element_class = (BinaryChromosome // 8, FloatChromosome // 2)
    def mutate(self):
        # Mutation based on the second chromosome
    def cross(self, other):
        # Crossover based on the second chromosome
    def _fitness(self):
        # Get fitness only depends on the first chromosome
        return f(self[0])
    
class NewPopulation(StandardPopulation):
    element_class = NewIndividual
    default_size = 20

pop = NewPopulation.random()
```

`FloatChromosome` comes pre-equipped with genetic operations tailored for floating-point numbers, obviating the necessity for user-defined specifications. This configuration facilitates the creation of GAs where mutation and crossover rates dynamically evolve.


## Comparison with other frameworks

Various genetic algorithm frameworks have been designed, such as deap and gaft. `Pyrimidine`'s design is heavily influenced by these frameworks. The following table compares `pyrimidine` with several popular and mature frameworks:

| Library   | Design Style      | Versatility | Extensibility | Visualization           |
| --------- | ------------------ | ---------- | ------------- | ---------------------- |
| pyrimidine| Object-Oriented, Metaprogramming, Algebraic-insprited | Universal | Extensible | export the data in `DataFrame` |
| deap      | Object-Oriented, Functional, Metaprogramming        | Universal | Extensible      | export the data in `LogBook`  |
| gaft      | Object-Oriented, decoration partton   | Universal | Extensible    | Easy to Implement       |
| tpot(gama)     | scikit-learn Style | Hyperparameter Optimization | Limited | None                   |
| gplearn   | scikit-learn Style | Symbolic Regression | Limited | None                   |
| scikit-opt| scikit-learn Style | Numerical Optimization | Limited | Encapsulated as a data frame      |

`tpot`, `gplearn`, and `scikit-opt` follow the `scikit-learn` style, providing fixed APIs with limited extensibility. However, they are mature and user-friendly, serving their respective fields effectively.

`deap` is feature-rich and mature. However, it primarily adopts a functional programming style. Some parts of the source code lack sufficient decoupling, limiting its extensibility. `gaft` is highly object-oriented with good extensibility, but not active. In `pyrimidine`, various operations on chromosomes are treated as chromosome methods, rather than top-level functions. When users customize chromosome operations, they only need to inherit the base chromosome class and override the corresponding methods. For example, the crossover operation for the `ProbabilityChromosome` class can be redefined as follows, suitable for optimization algorithms where variables follow a probability distribution:

```python
def cross(self, other):
    k = randint(1, len(self)-2)
    array = np.hstack((self[:k], other[k:]))
    array /= array.sum()
    return self.__class__(array)
```


## Conclusion

I have conducted extensive experiments and improvements, demonstrating that `pyrimidine` is a versatile framework suitable for implementing various evolution algorithms. Its design offers strong extensibility, allowing the implementation of any iterative model, such as simulated annealing or particle swarm optimization. For users developing new algorithms, `pyrimidine` is a promising choice.

`Pyrimidine` requires fitness values to be numbers, so it cannot handle multi-objective problems directly, unless they are reduced to single-objective problems. `Pyrimidine` uses numpy's arrays, making crossover operations slower than deap's, but alternative implementations can be used. Of course, there are other areas for improvement. Additionally, `pyrimidine`'s documentation is still under development.

The complete source code has been uploaded to GitHub, including numerous examples.https://github.com/Freakwill/pyrimidine.


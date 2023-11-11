---
title: 'Pyrimidine: Algebra-inspired Programming framework for genetic algorithms
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
Genetic algorithm (GA) is a general optimization method that mimics natural selection in evolutionary biology to solve optimization problems. It is the earliest developed intelligent algorithm [1-4], which has been widely used in multiple fields and has been modified and combined with new algorithms [5-6]. This article does not review its principles. Please refer to reference [4] and the literature cited therein for more information.

Currently, lots of programming languages provide libraries that implement GA frameworks. Python may provides most GA frameworks, including well-known libraries such as deap [7], gaft and tpot as the parameter optimizor[8-9] for machine learning, as well as scikit-opt and gplearn as the extensions of scikit-learn [10], etc. This article introduces pyrimidine, which is a general algorithm framework designed by the author. It strictly follows object-oriented programming(OOP) principles compared to other libraries and utilizes Python's metaprogramming capabilities.

## Algebra-inspired Programming

GA consists of two main components: individuals(choromosomes) and populations.

A oridinary idea of implementation in Python, populations are desienged as the lists of individuals, individuals( chromosomes) are the lists of genes. You can create an individual using the standard library `array` or the well-known third-party numerical computing library `numpy`[11]. The latter is convenient for various numerical computations but may be slower for cross operations.

Our design concept is beyond the oridinary idea and more extensible. We call it "algebra-inspired Programming".


### Metaclasses
The metaclass `System` simulates abstract algebraic systems, which are instantiated as a set containing a set of elements, as well as operations and functions on them.

`Container` is a sub-metaclass of `System`, where the elements has a certain type. The instance of `Container` is seen to be as subset of an algebraic system.

There are mainly two types of containers: list-like and tuple-like, where lists imply that all elements in the container are of the same type, while tuples have no such restriction.


### Fundamental Classes

The base classes mentioned in the paper are all constructed by the metaclasses. 

There are three fundamental classes in `pyrimidine`,`BaseChromosome`, `BaseIndividual`, `BasePopulation`, to create chromosomes, individuals and populations respectively. 

an individual is a container of multiple chromosomes, that is different from normal designs of GA, representing a single solution of some problem. a population is a container of individuals. In fact, a chromosome is designed as a container of genes, that is not so significant.

Constructing an individual, `SomeIndividual`, consisting of several `SomeChromosome`, is as simple as defining `SomeIndividual = BaseIndividual[SomeChromosome]`, where `BaseIndividual` is the base class for all individual classes. The expression is borrowed from variable typing, such as `List[String]`. Similarly, a population of `SomeIndividual` could be `BasePopulation[SomeIndividual]`.

A population is a container of individuals. It is called the high-order container.

For convenience, `pyrimidine` provides some commonly used subclasses, so users do not have to redefine these settings. By inheriting these classes, users gain access to the methods such as, cross and mutation. Genetic algorithms generally use binary encoding. `pyrimidine` offers `BinaryChromosome` for the binary settings. By inheriting from this class, users have binary encoding and algorithm components for two-point cross and independent gene mutation.

Generally, users start the algorithm design as follows, where `MonoIndividual` enforces that individuals can only have one chromosome (not mandatory).

```python
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome
    def _fitness(self):
        # Write the fitness calculation process here
```

`MyIndividual` is a class (an concrete container) with `BinaryChromosome` as its chromosome type. As you saw, it is equivalent to `MyIndividual=MonoIndividual[BinaryChromosome]`. Since `MonoBinaryIndividual` is such a class, an equivalent way to write this is,

```python
class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        ...
```

By using a standard GA as the iteration method, you can directly set `MyPopulation = StandardPopulation[MyIndividual]`. It's implemented through metaprogramming and signifies that `MyPopulation` is a list composed of `MyIndividual` objects.

There is no different between `MonoIndividual` and a single `Chromosome`. Actually, `Chromosome` class are used to simulate GA, not merely `Individual` class. And the population also can be a container of chromosomes.

```python
class MyChromosome(BaseChromosome):
    def _fitness(self):
        # Write the fitness calculation process here
```


<!-- ### Main methods
If you want to redesign the crossover and mutation in the algorithms, you can override the `mutate` and `cross` methods of the individual class. You can also override the methods of the chromosome class, since the operations of the individual class simply invoke the methods of the chromosome class. -->

### Mixin classes
These classes also inherit from  the mixin class `IterativeModel`, that is responsible for the iteration, including exporting data for visualization. When developing a new algorithm, the key point is to override the `transition` method, which is called in all iterative algorithms. For genetic algorithms, `transition` is mainly composed of `mutate` and `cross`, the crossover and the mutation methods.

As the subclass of `IterativeModel`, `FitnessModel` is cretated for executing the iterative algorithm to maximize the fitness.

### Mathematical representation
The auther use the following expression to represent a container $s$ of type $S$, with the elements of type $A$:
$$
s=\{a:A\}:S
$$
where $\{\cdot\}$ represets a set, or a sequence to emphasis the order of the elements.

As you saw, the population is a container of individuals. It is not difficult to generalize the concept to multi-population, which is the container of populations, and called the high-level container.

The lifting of a method $f$ of $a$ is defined as
$$
f(s):=\{f(a)\}
$$
unless it is redefined explictly. For example, the mutation of a population is the mutation of all indiviudals in it, but sometimes it may be defined as the mutation of one individual selected randomly.

Some methods would be lifted in other manners, such as
$$
f(s):=\max_t\{f(t)\}
$$
The main example is `fitness` to compute the fitness of the whole population.

As mensioned above, the `transition` transform is the most important method of $s$, denoted as
$$
T(s):S\to S
$$
Then an iteration is represented as $T^n(s)$.

With the help of the above concepts, we can impliment any iterative algorithm for socalled swarm intelligence.


## An Example to start

We have provided dozens of examples. In this section, we illustrate the basic usage of `pyrimidine` with a simple example: the classic 0-1 knapsack problem with $n=50$ dimensions:

$$
\max \sum_i c_ix_i \\
\sum_i w_ix_i \leq W, \quad x_i=0,1
$$

The solution of the problem can be naturally encoded in binary format without further decoding.

### Algorithm Construction

Using the classes provided by `pyrimidine`, it is straightforward to construct a population with 20 individuals, each containing a 50-dimensional chromosome. The population will iterate 100 times, and the fittest individual in the last generation will represent the solution to the optimization problem.

```python
from pyrimidine import MonoBinaryIndividual, StandardPopulation
from pyrimidine.benchmarks.optimization import Knapsack
# The problem is defined in the submodule `pyrimidine.benchmarks.optimization`, so we import it directly, but user can redefine it manually.

n = 50
_evaluate = Knapsack.random(n)  # Function mapping n-dimensional binary encoding to the objective function value

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        # The return value must be a number
        return _evaluate(self.chromosome)

"""
equivalent to:
MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))
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

Finally, the optimal individual can be obtained with `pop.best_individual`. The equivalent code below achieves the same result.

```python
pop = (StandardPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))] // 20).random()
pop.evolve()
```

The equivalent approach no longer explicitly depends on class inheritance and `class` syntax, making the code more concise and similar to algebraic operations.

### Visualization

To evaluate the performance of the algorithms, it is common to plot fitness curves or other metrics against the iteration number. This can be achieved by setting the `history=True` parameter in the `evolve` method. This method will return a `pandas.DataFrame` object containing statistical results for each generation. Users can use this object to freely plot performance curves. Typically, users need to provide a "statistic dictionary": keys are the names of the statistics, and values are functions that take the population and return numerical values (strings are limited to pre-defined population methods or attributes of return values). The statistical task is done in the mixin class `IterativeModel`, as well as the iteration.

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

Here `mean_fitness` and `best_fitness` represent the average fitness value of the population and the optimal individual fitness value respectively. Of course, they eventually have functions to achieve statistical functions, such as `best_fitness` corresponds to the mapping `pop->pop.best_individual.fitness`.

![](/Users/william/Programming/myGithub/pyrimidine/plot-history.png)


## Algorithm Extension

Developing new algorithms with `pyrimidine` is remarkably simple. 

### Create your own individual and GA
In classical GAs, the mutation rate and crossover rate are invariant and uniform across the entire population through the evolution. However, `pyrimidine` easily allows these rates to be encoded in each individual, making them adaptive during iterations.

We define a `MixedIndividual` with two chromosomes, one represents the solutions and the other represents the probability of mutation and the probability of crossover.

```python
class NewIndividual(MixedIndividual):
    element_class = (BinaryChromosome, FloatChromosome)
    def mutate(self):
        # Mutation based on the second chromosome
    def cross(self, other):
        # Crossover based on the second chromosome
    def _fitness(self):
        # Fitness only depends on the first chromosome
        f(self[0])
    
class NewPopulation(StandardPopulation):
    element_class = NewIndividual
    default_size = 20

# 8 represents the coding length for variables, 2 represents mutation rate and crossover rate
pop = NewPopulation.random(sizes=(8, 2))
```

`FloatChromosome` is already equipped with genetic operations for floating-point numbers, eliminating the need for user definition. This setup enables the development of a genetic algorithm where mutation and crossover rates evolve dynamically.

### Design new algorithm
Take PSO as example.

## Comparison

Various genetic algorithm frameworks have been designed, with deap and gaft being among the most popular ones. Pyrimidine's design is heavily influenced by these two frameworks, even borrowing class names directly from gaft. The following table compares pyrimidine with several popular and mature frameworks:

| Library   | Design Style      | Generality | Extensibility | Visualization           |
| --------- | ------------------ | ---------- | ------------- | ---------------------- |
| pyrimidine| Object-Oriented, Metaprogramming, Algebraic | Universal | Extensible | Easy to Implement, Customizable |
| deap      | Functional, Metaprogramming        | Universal | Limited       | Easy to Implement       |
| gaft      | Object-Oriented    | Universal | Extensible    | Easy to Implement       |
| tpot      | scikit-learn Style | Hyperparameter Optimization | Limited | None                   |
| gplearn   | scikit-learn Style | Symbolic Regression | Limited | None                   |
| scikit-opt| scikit-learn Style | Numerical Optimization | Limited | Easy to Implement      |

tpot, gplearn, and scikit-opt follow the scikit-learn style, providing fixed APIs with limited extensibility. However, they are mature and user-friendly, serving their respective fields effectively.

deap was the first genetic algorithm framework I started using. It is feature-rich and mature. However, it primarily adopts a functional programming style. Some parts of the source code lack sufficient decoupling, limiting its extensibility. gaft is highly object-oriented with good extensibility. The design approach in pyrimidine is slightly different from gaft. In pyrimidine, various operations on chromosomes are treated as chromosome methods, rather than independent functions. This design choice might not necessarily increase program coupling. When users customize chromosome operations, they only need to inherit the base chromosome class and override the corresponding methods. For example, the crossover operation for the ProbabilityChromosome class can be redefined as follows, suitable for optimization algorithms where variables follow a probability distribution:

```python
def cross(self, other):
    # Ensure that the sum of all genes (real numbers) in the chromosome is 1
    k = randint(1, len(self)-2)
    array = np.hstack((self[:k], other[k:]))
    array /= array.sum()
    return self.__class__(array=array, gene=self.gene)
```

## Conclusion

I have conducted extensive experiments and improvements, demonstrating that pyrimidine is a versatile framework suitable for implementing various genetic algorithms. Its design offers strong extensibility, allowing the implementation of any iterative model, such as simulated annealing or particle swarm optimization. For users developing new algorithms, pyrimidine is a promising choice.

Pyrimidine requires fitness values to be numbers, so it cannot handle multi-objective problems directly, unless they are reduced to single-objective problems. Pyrimidine uses numpy's arrays, making crossover operations slower than deap's, but alternative implementations can be used. Of course, there are other areas for improvement. Additionally, pyrimidine's documentation is still under development.

The complete source code has been uploaded to GitHub, including numerous examples (see the "examples" folder).https://github.com/Freakwill/pyrimidine。

<!-- Currently, operating the individual could not access. -->


---
<center>References</center>
[1] Holland, J. Adaptation in Natural and Artificial Systems[M]. The Univ. of Michigan, 1975.
[3] D. Simon. 进化优化算法——基于仿生和种群的计算机智能方法[M]. 北京: 清华大学出版社, 2018.
[4] 玄光男， 程润伟. 遗传算法与工程优化[M]. 北京: 清华大学出版社, 2004.
[7]Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, DEAP: Evolutionary Algorithms Made Easy[J]. Journal of Machine Learning Research, 2012, 13: 2171-2175.
[8] Randal S. Olson, Ryan J. Urbanowicz, Peter C. Andrews, Nicole A. Lavender, La Creis Kidd, and Jason H. Moore. Automating biomedical data science through tree-based pipeline optimization[J]. Applications of Evolutionary Computation, 2016: 123-137.
[9] Trang T. Le, Weixuan Fu and Jason H. Moore. Scaling tree-based automated machine learning to biomedical big data with a feature set selector. Bioinformatics[J], 2020, 36(1): 250-256.
[10] Scikit-learn https://scikit-learn.org/[OL].
[11]Numpy. https://numpy.org/[OL].
[13] typing — Support for type hints. https://docs.python.org/3.8/library/typing.html?highlight=typing#module-typing[OL]


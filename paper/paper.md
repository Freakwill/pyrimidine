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

# Pyrimidine: Algebra-inspired Programming framework for evolution algorithms

**Abstract** Pyrimidine is a general framework for evolution algorithms. It is extremely extensible and can implement any iterative model, such as simulated annealing and particle swarm optimization. Its design is based on object-oriented programming and fully utilizes the metaprogramming capabilities of Python. We propose a container metaclass to construct different structures of individual and population classes. These classes are understood as algebraic systems, where elements can perform various operations, such as mutation and crossover of individuals in a population. These classes may also be the elements of higher-order classes, allowing for automatic implementation of class-level operations such as population migration in evolution algorithms. We call such design style "the algebra-inspired Progamming".


**Keywords** Pyrimidine, Evolution Algorithms, Algebra-inspired Programming, Python, meta-programming

## Introduction
Initially developed as a general algorithm(GA), Pyrimidine has evolved to accommodate various types of evolutionary algorithms.

As one of the earliest developed intelligent algorithms [1-4], GA has found extensive application across various domains and has undergone modifications and integrations with new algorithms [5-6]. The principles of GA will not be extensively reviewed in this article. For a detailed understanding, please refer to reference [4] and the associated literature.

Presently, a variety of programming languages feature libraries that implement Genetic Algorithm (GA) frameworks. Python stands out for its extensive collection of GA frameworks, including notable ones like deap [7] for general purposes, gaft for optimization, and tpot for super-parameter tuning [8-9], along with scikit-learn, such as scikit-opt and gplearn [10]. This article introduces pyrimidine, a general algorithm framework for GA and other EA. Adhering rigorously to object-oriented programming (OOP) principles, pyrimidine distinguishes itself from other libraries, making effective use of Python's metaprogramming capabilities.

## Algebra-inspired Programming

As known, GA consists of two main components: individuals( or choromosomes) and populations.

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

### Mathematical representation
We use the following expression to represent a container $s$ of type $S$, with elements of type $A$:
$$
s = \{a:A\}:S
$$
where $\{\cdot\}$ represents a set or a sequence to emphasize the order of the elements.

As observed, the population is a container of individuals. Generalizing this concept to multi-population is straightforward, where it becomes the container of populations and is referred to as the high-level container.

The lifting of a method $f$ of $a$ is defined as:
$$
f(s) := \{f(a)\}
$$
unless explicitly redefined. For instance, the mutation of a population entails the mutation of all individuals in it, but at times, it may be defined as the mutation of one individual selected randomly.

Some methods may be lifted differently, such as:
$$
f(s) := \max_t\{f(t)\}
$$
A notable example is `fitness`, used to compute the fitness of the entire population.

As mentioned earlier, the `transition` transform is the primary method in the iterative algorithms, denoted as:
$$
T(s):S\to S
$$
Consequently, the iteration can be represented as $T^n(s)$.
And if $s$ is a container, then $T(s)=\{T(a)\}$ by default where $T(a)$ is pre-defined.

## An Example to start

In this section, we demonstrate the fundamental usage of `pyrimidine` through a simple illustration: the classic 0-1 knapsack problem with $n=50$ dimensions. (See more examples on GitHub)

$$
\max \sum_i c_ix_i \\
\sum_i w_ix_i \leq W, \quad x_i=0,1
$$

The problem solution can be naturally encoded in binary format without requiring additional decoding.


```python
from pyrimidine import MonoIndividual, StandardPopulation
from pyrimidine.benchmarks.optimization import Knapsack
# The problem is defined in the submodule `pyrimidine.benchmarks.optimization`, so we import it directly, but user can redefine it manually.

n = 50
_evaluate = Knapsack.random(n)  # Function mapping n-dimensional binary encoding to the objective function value

class MyIndividual(MonoIndividual):
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

Finally, the optimal individual can be obtained with `pop.best_individual` or `pop.solution` as the solution of the problem. The equivalent code below achieves the same result.

```python
pop = (StandardPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))] // 20).random()
pop.evolve()
```

The equivalent approach no longer explicitly depends on class inheritance and `class` syntax, making the code more concise and similar to algebraic style.

## Visualization

To assess the performance, it is customary to visualize fitness curves or other metrics against the iteration number. This can be accomplished by enabling the `history=True` parameter in the `evolve` method. Subsequently, this method yields a `pandas.DataFrame` object that encapsulates statistical results for each generation. Users can harness this object to create customizable performance curves. Generally, users are required to furnish a "statistic dictionary," where keys are the names of the statistics, and values are functions mapping the population to numerical values (strings are confined to pre-defined population methods or attributes of return values).

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
In classical GAs, the mutation rate and crossover rate remain constant and uniform throughout the entire population during evolution. However, in self-adaptive GA, these rates can be dynamically encoded within each individual, allowing for adaptability during iterations. It is remarkably simple to implement self-adaptive GA by `pyrimidine`. 

We introduce a `MixedIndividual` comprising two chromosomes—one representing the solutions and the other encapsulating the probabilities of mutation and crossover.

```python
class NewIndividual(MixedIndividual):
    element_class = (BinaryChromosome // 8, FloatChromosome // 2)
    def mutate(self):
        # Mutation based on the second chromosome
    def cross(self, other):
        # Crossover based on the second chromosome
    def _fitness(self):
        # Get fitness only depends on the first chromosome
        f(self[0])
    
class NewPopulation(StandardPopulation):
    element_class = NewIndividual
    default_size = 20

pop = NewPopulation.random()
```

`FloatChromosome` comes pre-equipped with genetic operations tailored for floating-point numbers, obviating the necessity for user-defined specifications. This configuration facilitates the creation of GAs where mutation and crossover rates dynamically evolve.



## Comparison with other frameworks

Various genetic algorithm frameworks have been designed, with deap and gaft being among the most popular ones. Pyrimidine's design is heavily influenced by these two frameworks, even borrowing class names directly from gaft. The following table compares pyrimidine with several popular and mature frameworks:

| Library   | Design Style      | Generality | Extensibility | Visualization           |
| --------- | ------------------ | ---------- | ------------- | ---------------------- |
| pyrimidine| Object-Oriented, Metaprogramming, Algebraic-insprited | Universal | Extensible | export the data in `DataFrame` |
| deap      | Functional, Metaprogramming        | Universal | Limited       | export the data in `LogBook`  |
| gaft      | Object-Oriented, decoration partton   | Universal | Extensible    | Easy to Implement       |
| tpot(gama)     | scikit-learn Style | Hyperparameter Optimization | Limited | None                   |
| gplearn   | scikit-learn Style | Symbolic Regression | Limited | None                   |
| scikit-opt| scikit-learn Style | Numerical Optimization | Limited | Easy to Implement      |

tpot, gplearn, and scikit-opt follow the scikit-learn style, providing fixed APIs with limited extensibility. However, they are mature and user-friendly, serving their respective fields effectively.

deap is feature-rich and mature. However, it primarily adopts a functional programming style. Some parts of the source code lack sufficient decoupling, limiting its extensibility. gaft is highly object-oriented with good extensibility, but not active. The design approach in pyrimidine is slightly different from gaft. In pyrimidine, various operations on chromosomes are treated as chromosome methods, rather than independent functions. This design choice might not necessarily increase program coupling. When users customize chromosome operations, they only need to inherit the base chromosome class and override the corresponding methods. For example, the crossover operation for the ProbabilityChromosome class can be redefined as follows, suitable for optimization algorithms where variables follow a probability distribution:

```python
def cross(self, other):
    k = randint(1, len(self)-2)
    array = np.hstack((self[:k], other[k:]))
    array /= array.sum()
    return self.__class__(array=array, gene=self.gene)
```

## Conclusion

I have conducted extensive experiments and improvements, demonstrating that pyrimidine is a versatile framework suitable for implementing various evolution algorithms. Its design offers strong extensibility, allowing the implementation of any iterative model, such as simulated annealing or particle swarm optimization. For users developing new algorithms, pyrimidine is a promising choice.

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


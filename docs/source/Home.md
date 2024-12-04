# pyrimidine: a universal framework for genetic algorithm

It is a highly object-oriental implement of genetic algorithm by Python.

![LOGO](source/green_logo.png)


## Why

Why is the package named as "pyrimidine"? Because it begins with "py". 

--- Are you kiding? 

--- No, I am serious.

## Download

It has been uploaded to pypi, so download it with `pip install pyrimidine`, and also could download it from github.

## Idea of algebra-inspired
We view the population as a container of individuals, each individual as a container of chromosomes, and a chromosome as a container (array) of genes. This container could be represented as a list or an array. The Container class has an attribute `element_class`, which specifies the class of the elements within it.

Mathematically, we denote a container of elements of type `A` as follows:

```
s = {a:A}:S  <==> s: S[A]
```

Now we give the following definition:
- A population is a container of individuals (or chromosomes); 
- An individual is a container of chromosomes; 
- A multi-population is a container of populations;

Below is the partial source code for `BaseIndividual` and `BasePopulation`.

```python
class BaseIndividual(FitnessMixin, metaclass=MetaContainer):
    element_class = BaseChromosome
    default_size = 1
    
class BasePopulation(FitnessMixin, metaclass=MetaHighContainer):
    element_class = BaseIndividual
    default_size = 20
```

where `FitnessMixin` is a mixin, representing a [iterative algorithm](https://pyrimidine.readthedocs.io/en/latest/source/API%20Design.html#iterative-models) with fitness.

There are mainly two kinds of containers: list and tuple as in programming language `Haskell`. See following examples.

```python
# individual with chromosomes of type _Chromosome
_Individual1 = BaseIndividual[_Choromosome]
# individual with 20 chromosomes of type _Chromosome
_Individual1 = BaseIndividual[_Choromosome] // 20
# individual with 2 chromosomes of type _Chromosome1 and _Chromosome2 respectively
_Individual2 = MixedIndividual[_Chromosome1, _Chromosome2]
```

A population can also serve as a container of chromosomes, particularly in scenarios where an individual possesses only a single chromosome.

In essence, a container - and by extension, a population in genetic algorithms - is regarded as a distinctive algebraic system. This perspective leads us to refer to it as an "algebra-inspired" design.

## Fitness

This is how we compute `fitness`. The method `_fitness` is responsible for the underlying computation. The attribute `fitness` further encapsulates `_fitness`. If caching is enabled, it will first read from the cache; if not, it will call `_fitness`.

It is recommended to add the `@fitness_cache` decorator to individuals. If the individual has not changed, then it can reduce computation and improve algorithm efficiency, otherwise it should re-compute fitness. The shortage is that you have to add `@side_effect` to the methods which have side effect, namly changing the fitness of the individual.

Unlike the cache class decorator, the `memory` decorator (e.g., `@basic_memory`) will change the algorithm's behavior. It stores the best results during the individual's changes. `fitness` will first read from memory. Memory itself also has a caching effect, so if you add the memory decorator, there is no need to add the cache decorator.

## Statement of need

As one of the earliest developed optimization algorithms [@holland; @katoch], the genetic algorithm (GA) has found extensive application across various domains and has undergone modifications and integrations with new algorithms [@alam; @cheng; @katoch]. The principles of GA will not be reviewed in this article. For a detailed understanding, please refer to references [@holland; @simon] and the associated literatures.

In a typical Python implementation, populations are initially defined as lists of individuals, with each individual represented by a chromosome composed of a list of genes. Creating an individual can be achieved utilizing either the standard library's `array` or the widely-used third-party library [`numpy`](https://numpy.org/) [@numpy]. The evolutionary operators are defined and applied to these structures.

A concise comparison between `pyrimidine` and other frameworks is provided in \autoref{frameworks}, such as [`DEAP`](https://deap.readthedocs.io/) [@fortin] and [`gaft`](https://github.com/PytLab/gaft), which have significantly influenced the design of `pyrimidine`.

<!-- +-------------------+------------+----------+----------+----------+ -->
| Library   | Design Style      | Versatility | Extensibility | Visualization           |
|:----------:|:-------|:--------|:--------|:----------|
| `pyrimidine`| OOP, Meta-programming, Algebra-insprited | Universal | Extensible | export the data in `DataFrame` |
| `DEAP`     | OOP, Functional, Meta-programming        | Universal | Limited by its philosophy   | export the data in the class `LogBook`  |
| `gaft`      | OOP, decoration pattern   | Universal | Extensible    | Easy to Implement       |
| [`geppy`](https://geppy.readthedocs.io/) | based on `DEAP` | Symbolic Regression | Limited | - |
| [`tpot`](https://github.com/EpistasisLab/tpot) /[`gama`](https://github.com/openml-labs/gama)  | [scikit-learn](https://scikit-learn.org/) Style | Hyperparameter Optimization | Limited | -                   |
| [`gplearn`](https://gplearn.readthedocs.io/)/[`pysr`](https://astroautomata.com/PySR/)   | scikit-learn Style | Symbolic Regression | Limited | -                  |
| [`scikit-opt`](https://github.com/guofei9987/scikit-opt)| scikit-learn Style | Numerical Optimization | Unextensible | Encapsulated as a data frame      |
|[`scikit-optimize`](https://scikit-optimize.github.io/stable/)|scikit-learn Style  | Numerical Optimization | Very Limited | provide some plotting function |
|[`NEAT`](https://neat-python.readthedocs.io/) | OOP  | Neuroevolution | Limited | use the visualization tools |

: Comparison of the popular genetic algorithm frameworks. \label{frameworks}

`Tpot`/`gama`[@olson; @pieter], `gplearn`/`pysr`, and `scikit-opt` follow the scikit-learn style [@sklearn_api], providing fixed APIs with limited extensibility. They are merely serving their respective fields effectively (as well as `NEAT`[@neat-python]).

`DEAP` is feature-rich and mature. However, it primarily adopts a tedious meta-programming style. Some parts of the source code lack sufficient decoupling, limiting its extensibility. `Gaft` is a highly object-oriented software with excellent scalability, but is currently inactive.

`Pyrimidine` fully utilizes the OOP and meta-programming capabilities of Python, making the design of the APIs and the extension of the program more natural. So far, we have implemented a variety of optimization algorithms by `pyrimidine`, including adaptive GA [@hinterding], quantum GA [@supasil], differential evolution [@radtke], evolutionary programming [@fogel], particle swarm optimization [@wang], as well as some local search algorithms, such as simulated annealing [@kirkpatrick].

To meet diverse demands, it provides enough encoding schemes for solutions to optimization problems, including Boolean, integer, real number types and their hybrid forms.

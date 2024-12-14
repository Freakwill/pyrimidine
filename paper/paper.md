---
title: 'Pyrimidine: An algebra-inspired Programming framework for evolutionary algorithms'
tags:
  - Python
  - genetic algorithms
  - evolutionary algorithms
  - intelligent algorithms
  - algebraic system
  - meta-programming
authors:
  - name: Congwei Song
    orcid: 0000-0002-4409-7276
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Beijing Institute of Mathematical Sciences and Applications, Beijing, China
   index: 1
date: 12 December 2023
bibliography: paper.bib
toccolor: teal
citecolor: teal
linkcolor: teal
urlcolor: teal
output: 
  pdf_document: 
    toc: no
    toc_depth: 4
    number_sections: yes
---

# `Pyrimidine`: An algebra-inspired Programming framework for evolutionary algorithms

# Summary

[`Pyrimidine`](https://github.com/Freakwill/pyrimidine) stands as a versatile framework designed for genetic algorithms (GAs), offering exceptional extensibility for a wide array of evolutionary algorithms.

Leveraging the principles of object-oriented programming (OOP) and the meta-programming, we introduce a distinctive design paradigm is coined as "algebra-inspired Programming" signifying the fusion of algebraic methodologies with the software architecture.

# Statement of need

GAs [@holland; @katoch] have found extensive application across various domains and have undergone modifications and integrations with new algorithms [@alam; @cheng; @katoch]. For details about the principles of GA, refer to the references [@holland; @simon].

In a typical Python implementation, populations are defined as lists of individuals, with each individual represented by a chromosome composed of a list of genes. Creating an individual can be achieved utilizing either the standard library's `array` or the widely-used third-party library [`numpy`](https://numpy.org/) [@numpy]. The evolutionary operators are defined on these structures.

A concise comparison between `pyrimidine` and other frameworks is provided in \autoref{frameworks}.

<!-- +-------------------+------------+----------+----------+----------+ -->
| Library   | Design Style      | Versatility | Extensibility | Visualization           |
|:----------:|:-------|:--------|:--------|:----------|
| `pyrimidine`| OOP, Meta-programming, Algebra-insprited | Universal | Extensible | export the data in `DataFrame` |
| [`DEAP`](https://deap.readthedocs.io/) | OOP, Functional, Meta-programming        | Universal | Limited by its philosophy   | export the data in the class `LogBook`  |
| [`gaft`](https://github.com/PytLab/gaft) | OOP, decoration pattern   | Universal | Extensible    | Easy to Implement       |
| [`geppy`](https://geppy.readthedocs.io/) | based on `DEAP` | Symbolic Regression | Limited | - |
| [`tpot`](https://github.com/EpistasisLab/tpot) /[`gama`](https://github.com/openml-labs/gama)  | [scikit-learn](https://scikit-learn.org/) Style | Hyperparameter Optimization | Limited | -                   |
| [`gplearn`](https://gplearn.readthedocs.io/)/[`pysr`](https://astroautomata.com/PySR/)   | scikit-learn Style | Symbolic Regression | Limited | -                  |
| [`scikit-opt`](https://github.com/guofei9987/scikit-opt)| scikit-learn Style | Numerical Optimization | Unextensible | Encapsulated as a data frame      |
|[`scikit-optimize`](https://scikit-optimize.github.io/stable/)|scikit-learn Style  | Numerical Optimization | Very Limited | provide some plotting function |
|[`NEAT`](https://neat-python.readthedocs.io/) | OOP  | Neuroevolution | Limited | use the visualization tools |

: Comparison of the popular genetic algorithm frameworks. \label{frameworks}

`Tpot`/`gama`[@olson; @pieter], `gplearn`/`pysr`, and `scikit-opt` follow the scikit-learn style [@sklearn_api], providing fixed APIs with limited extensibility. They are merely serving their respective fields effectively (including `NEAT`[@neat-python]).

`DEAP`[@fortin] is feature-rich and mature. However, it adopts a tedious meta-programming style and some parts of the code lack decoupling, limiting its extensibility. `Gaft` is highly object-oriented and scalable, but inactive now.

`Pyrimidine` fully utilizes the OOP and meta-programming capabilities of Python, making the design of the APIs and the extension of the program more natural. So far, we have implemented a variety of optimization algorithms by `pyrimidine`, including adaptive GA [@hinterding], quantum GA [@supasil], differential evolution [@radtke], evolutionary programming [@fogel], particle swarm optimization [@wang], as well as some local search algorithms, such as simulated annealing [@kirkpatrick].

To meet diverse demands, it provides enough encoding schemes for solutions to optimization problems, including Boolean, integer, real number types and their hybrid forms.

# Algebra-inspired programming

The innovative approach is termed "algebra-inspired Programming". It should not be confused with so-called algebraic programming [@kapitonova], but it draws inspiration from its underlying principles.

The advantages of the model are summarized as follows:

1. The population system and genetic operations are treated as an algebraic system, and genetic algorithms are constructed by imitating algebraic operations.
2. It is highly extensible. For example it is easy to define multi-populations, even so-called hybrid-populations.
3. The code is more concise.

## Basic concepts

We introduce the concept of a **container**, simulating an **(algebraic) system** where specific operators are not yet defined.

A container $s$ of type $S$, with elements of type $A$, is represented by the following expression:
\begin{equation}\label{eq:container}
s = \{a:A\}: S \quad \text{or} \quad s:S[A]\,,
\end{equation}
where the symbol $\{\cdot\}$ signifies either a set, or a sequence to emphasize the order of the elements. The notation $S[A]$ mimicks Python syntax, borrowed from the module [typing](https://docs.python.org/3.11/library/typing.html?highlight=typing#module-typing).

Building upon the concept, we define a population in `pyrimidine` as a container of individuals. The introduction of multi-population further extends this notion, representing a container of populations, referred to as "the high-order container". `Pyrimidine` distinguishes itself with its inherent ability to seamlessly implement multi-population GAs.

An individual is conceptualized as a container of chromosomes, without necessarily being an algebraic system. Similarly, a chromosome acts as a container of genes.

In a population system $s$, the formal representation of the crossover operation between two individuals is denoted as $a \times_s b$, that can be implemented as the command `s.cross(a, b)`. Although this system concept aligns with algebraic systems, the current version diverges from this notion, and the operators are directly defined as methods of the elements, such as `a.cross(b)`.

The lifting of a function/method $f$ is a common approach to defining the function/method for the system:
$$
f(s) := \{f(a)\}\,,
$$
unless explicitly redefined. For example, the mutation of a population typically involves the mutation of all individuals in it. Other types of lifting are allowed.

`transition` is the primary method in the iterative algorithms, denoted as a transform:
$$
T(s): S\to S\,.
$$

## Metaclasses

A metaclass should be defined to simulate abstract algebraic systems, which are instantiated as a set containing several elements, as well as operators and functions on them. Currently, the metaclass `MetaContainer` is proposed to create container classes without defining operators explicitly.

## Mixin classes

Mixin classes specify the basic functionality of the algorithm.

The `FitnessMixin` class is dedicated to the iteration process focused on maximizing fitness, and its subclass `PopulationMixin` represents the collective form.

When designing a novel algorithm, significantly differing from the GA, it is advisable to start by inheriting from the mixin classes.

## Base Classes

There are three base classes in `pyrimidine`: `BaseChromosome`, `BaseIndividual`, `BasePopulation`, to create chromosomes, individuals and populations respectively.

Generally, the algorithm design starts as follows.

```python
class UserIndividual(MonoIndividual):
    element_class = BinaryChromosome

    def _fitness(self):
        # Compute the fitness

class UserPopulation(StandardPopulation):
    element_class = UserIndividual
    default_size = 10
```

Here, `MonoIndividual` represents an individual with a single chromosome. `UserIndividual` (or `UserPopulation`) serves as a container of elements in type of `BinaryChromosome` (or `UserIndividual`). Instead of setting the `fitness` attribute, users are recommended to override the `_fitness` method, where the concrete fitness computation is defined. Following is the equivalent expression, using the notion in \autoref{eq:container}:

```python
UserIndividual = MonoIndividual[BinaryChromosome]
UserPopulation = StandardPopulation[UserIndividual] // 10
```

Algebraically, there is no difference between `MonoIndividual` and `Chromosome`. And the population also can be treated as a container of chromosomes as follows.

```python
class UserChromosome(BaseChromosome):
    def _fitness(self):
        # Compute the fitness

UserPopulation = StandardPopulation[UserChromosome] // 10
```

# Conclusion

`Pyrimidine` is a versatile framework suitable for implementing various evolution algorithms. Its design offers strong extensibility. A key factor is that it was developed inspired by algebra. For users developing novel algorithms, `pyrimidine` is a promising choice.

# References

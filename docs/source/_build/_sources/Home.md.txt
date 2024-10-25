# pyrimidine: a universal framework for genetic algorithm

It is a highly object-oriental implement of genetic algorithm by Python.

![LOGO](logo.png)


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
# pyrimidine: a universal framework for genetic algorithm

It is a highly object-oriental implement of genetic algorithm by Python.

![LOGO](logo.png)


## Why

Why is the package named as "pyrimidine"? Because it begins with "py". 

--- Are you kiding? 

--- No, I am serious.

## Download

It has been uploaded to pypi, so download it with `pip install pyrimidine`, and also could download it from github.

## Idea

We regard the population as a container of individuals, an individual as a container of chromosomes
and a chromosome as a container(array) of genes.

The container could be a list or an array. Container class has an attribute `element_class`, telling itself the class of the elements in it.

Mathematically, we denote a container of elements in type `A` as following
```
s = {a:A}:S
```

A population is a container of individuals; An indiviudal is a container of chromosomes. Following is the part of the source code of `BaseIndividual` and `BasePopulation`.

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

An population also could be the container of chromosomes. It will be considered in the case when the indiviudal has only one chromosome.

In fact, a container (so a population in GA) is treated as a special algebraic system. For this reason, we call it "algebra-oriental" design.

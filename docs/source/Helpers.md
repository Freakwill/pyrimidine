# Helpers

To introduce useful helpers

## Optimization

Why is the package named as "pyrimidine"? Because it begins with "py". 

--- Are you kiding? 

--- No, I am serious.

## Download

It has been uploaded to pypi, so download it with `pip install pyrimidine`, and also could download it from github.

## Idea

We regard the population as a container of individuals, an individual as a container of chromosomes
and a chromosome as a container(array) of genes.

The container could be a list or an array. Container class has an attribute `element_class`, telling itself the class of the elements in it.

Following is the part of the source code of `BaseIndividual` and `BasePopulation`.

```python
class BaseIndividual(FitnessModel, metaclass=MetaContainer):
    element_class = BaseChromosome
    default_size = 1
    
class BasePopulation(FitnessModel, metaclass=MetaHighContainer):
    element_class = BaseIndividual
    default_size = 20
```

where `FitnessModel` is a mixin, as a iteravtive model with fitness.

There are mainly two kinds of containers: list and tuple as in programming language `Haskell`. See following examples.

```python
# individual with chromosomes of type _Chromosome
_Individual1 = BaseIndividual[_Choromosome]
# individual with 20 chromosomes of type _Chromosome
_Individual1 = BaseIndividual[_Choromosome] // 20
# individual with 2 chromosomes of type _Chromosome1 and _Chromosome2 respectively
_Individual2 = MixedIndividual[_Chromosome1, _Chromosome2]
```



In fact, a container (so a population in GA) is treated as a special algebraic system. For this reason, we call it algebra-oriental design.

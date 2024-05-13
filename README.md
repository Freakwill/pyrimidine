# pyrimidine

[![Pytest](https://github.com/Freakwill/pyrimidine/actions/workflows/pytest.yml/badge.svg)](https://github.com/Freakwill/pyrimidine/actions/workflows/pytest.yml)
[
![PyPi](https://img.shields.io/pypi/v/pyrimidine.svg)
](https://pypi.python.org/pypi/pyrimidine)

`pyrimidine` is an extensible framework of genetic/evolutionary algorithm by Python. See [pyrimidine's document](https://pyrimidine.readthedocs.io/) for more details.

![LOGO](logo.png)

## Why

> -- Why is the package named as “pyrimidine”? 
> -- Because it begins with “py”.
> -- Are you kiding?
> -- No, I am serious.

If you have more questions, then log in [google group](https://groups.google.com/g/pyrimidine) and post your questions.

## Download

It has been uploaded to [pypi](https://pypi.org/project/pyrimidine/), so download it with `pip install pyrimidine`, and also could download it from Github.

## Video tutorials

[An gental introduction](https://www.youtube.com/watch?v=uVf3y427ei4&t=3s)

## Idea

We view the population as a container of individuals, each individual as a container of chromosomes, and a chromosome as a container (array) of genes. This container could be represented as a list or an array. The Container class has an attribute `element_class`, which specifies the class of the elements within it.

Following is the part of the source code of `BaseIndividual` and `BasePopulation`.

```python
class BaseIndividual(FitnessModel, metaclass=MetaContainer):
    element_class = BaseChromosome
    default_size = 1

class BasePopulation(PopulationModel, metaclass=MetaContainer):
    element_class = BaseIndividual
    default_size = 20
```

There is two main kinds of containers: list-like and tuple-like. See following examples.

```python
# individual with chromosomes of type _Chromosome
_Individual1 = BaseIndividual[_Choromosome]
# individual with 2 chromosomes of type _Chromosome1 and _Chromosome2 respectively
_Individual2 = MixedIndividual[_Chromosome1, _Chromosome2]
```

### Math expression

$s$ of type $S$ is a container of $a:A$, represented as follows:

```
s={a:A}:S
```

We could define a population as a container of individuals or chromosomes, and an individual is a container of chromosomes.

Algebraically, an indivdiual has only one chromosome is equivalent to a chromosome mathematically. A population could also be a container of chromosomes. If the individual has only one chromosome, then just build the population based on chromosomes directly.

The methods are the functions or operators defined on $s$.

## Use

### Main classes

- BaseGene: the gene of chromosome
- BaseChromosome: sequence of genes, represents part of a solution
- BaseIndividual: sequence of chromosomes, represents a solution of a problem
- BasePopulation: a container of individuals, represents a container of a problem
                also the state of a stachostic process
- BaseMultipopulation: a container of population for more complicated optimalization


### import

To import all algorithms for beginners, simply use the command `from pyrimidine import *`.

To speed the lib, use the following commands 
```
from pyrimidine import BaseChromosome, BaseIndividual, BasePopulation # import the base classes form `base.py` to build your own classes

# Commands used frequently
from pyrimidine.base import BinaryChromosome, FloatChromosome # import the Chromosome classes and utilize them directly
# equivalent to `from pyrimidine import BinaryChromosome, FloatChromosome`
from pyrimidine.population import StandardPopulation, HOFPopulation # For creating population with standard GA 
# the same effect with `from pyrimidine import StandardPopulation, HOFPopulation`
from pyrimidine.indiviual import makeIndividual # a helper to make Individual objects, or `from pyrimidine import makeIndividual`

from pyrimidine import MultiPopulation # build the multi-populations
from pyrimidine import MetaContainer # meta class for socalled container class, that is recommended to be used for creating novel evolutionary algorithms.

from pyrimidine.deco import * # import all decorators
from pyrimidine.deco import fitness_cache, basic_memory # use the cache decorator and memory decorator

from pyrimidine import optimize # do optimization implictly with GAs
```

### subclasses

#### Chromosome

Generally, it is an array of genes.

As an array of 0-1s, `BinaryChromosome` is used most frequently.

#### Individual
just subclass `MonoIndividual` in most cases.

```python
from pyrimidine.individual import MonoIndividual
from pyrimidine.chromosome import BinaryChromosome

# or from pyrimidine import MonoIndividual, BinaryChromosome

class MyIndividual(MonoIndividual):
    """individual with only one chromosome
    we set the gene is 0 or 1 in the chromosome
    """
    element_class = BinaryChromosome

    def _fitness(self):
        ...
```

Since the helper `makeIndividual(n_chromosomes=1, size=8)` could create such individual, it is equivalent to

```python
from pyrimidine.individual import binaryIndividual

class MyIndividual(binaryIndividual()):
    # only need define the fitness
    def _fitness(self):
        ...
```

If an individual contains several chromosomes, then subclass `MultiIndividual` or `PolyIndividual`. It could be applied in multi-real-variable optimization problems where each variable has a separative binary encoding.

In most cases, we have to decode chromosomes to real numbers.

```python
from pyrimidine.individual import BaseIndividual
from pyrimidine.chromosome import BinaryChromosome

class _Chromosome(BinaryChromosome):
    def decode(self):
        """Decode a binary chromosome
        
        if the sequence of 0-1 represents a real number, then overide the method
        to transform it to a nubmer
        """

class ExampleIndividual(BaseIndividual):
    element_class = _Chromosome

    def _fitness(self):
        # define the method to calculate the fitness
        x = self.decode()  # will call decode method of _Chromosome
        return evaluate(x)
```

If the chromosomes in an individual are different with each other, then subclass `MixedIndividual`, meanwhile, the property `element_class` should be assigned with a tuple of classes for each chromosome.

```python
from pyrimidine.individual import MixedIndividual

class MyIndividual(MixedIndividual):
    """
    Inherit the fitness from ExampleIndividual directly.
    It has 6 chromosomes, 5 are instances of _Chromosome, 1 is instance of FloatChromosome
    """
    element_class = (_Chromosome,)*5 + (FloatChromosome,)
```

It equivalent to `MyIndividual=MixedIndividual[(_Chromosome,)*5 + (FloatChromosome,)]`

#### Population

```python
from pyrimidine.population import StandardPopulation

class MyPopulation(StandardPopulation):
    element_class = MyIndividual
```

It is equivalent to `MyPopulation = StandardPopulation[MyIndividual]`.


### Initialize randomly

`random` is a factory method!

Generate a population, with 50 individuals and each individual has 100 genes:

`pop = MyPopulation.random(n_individuals=50, size=100)`

When each individual contains 5 chromosomes, use

`pop = MyPopulation.random(n_individuals=10, n_chromosomes=5, size=10)`

However, we recommand to set `default_size` in the classes, then run `MyPopulation.random()`

```python
from pyrimidine.population import StandardPopulation

class MyPopulation(StandardPopulation):
    element_class = MyIndividual // 5
    default_size = 10

# equiv. to

MyPopulation = StandardPopulation[MyIndividual//5]//10
```

In fact, `random` method of `BasePopulation` will call random method of `BaseIndividual`. If you want to generate an individual, then just execute `MyIndividual.random(n_chromosomes=5, size=10)`, or set `default_size`, then execute `MyIndividual.random()`.

### Evolution

#### `evolve` method
Initialize a population with `random` method, then call `evolve` method.

```python
pop = MyPopulation.random(n_individuals=50, size=100)
pop.evolve()
print(pop.solution)
```

set `verbose=True` to display the data for each generation.

`evolve` method mainly excute two methods: 
- `init`: initial configuration of the algo.
- `transition`: do each step of the iteration.

#### History

Get the history of the evolution.

```python
stat={'Mean Fitness':'mean_fitness', 'Best Fitness': lambda pop: pop.best_individual.fitness}
data = pop.history(stat=stat)  # use history instead of evolve
```
`stat` is a dict mapping keys to function, where string 'mean_fitness' means function `lambda pop:pop.mean_fitness` which gets the mean fitness of the individuals in `pop`. Since we have defined pop.best_individual.fitness as a property, `stat` could be redefined as `{'Fitness': 'fitness', 'Best Fitness': 'max_fitness'}`.

It requires `ezstat` (optional but recommended), a easy statistical tool devoloped by the author.

#### performance

Use `pop.perf()` to check the performance, which calls `evolve` several times.

## Example

### Example 1

Description

    select some of ti, ni, i=1,...,L, ti in {1,2,...,T}, ni in {1,2,...,N}
    the sum of ni approx. 10, while ti dose not repeat

The opt. problem is

    min abs(sum_i{ni}-10) + maximum of frequences in {ti}
    where i is selected.

$$
\min_I |\sum_{i\in I} n_i -10| + t_m
\\
I\subset\{1,\cdots,L\}
$$
where $t_m$ is the mode of $\{t_i, i\in I\}$

```python
import numpy as np

t = np.random.randint(1, 5, 100)
n = np.random.randint(1, 4, 100)

import collections

from pyrimidine.individual import makeBinaryIndividual
from pyrimidine.population import StandardPopulation


def max_repeat(x):
    # Maximum repetition
    c = collections.Counter(x)
    return np.max([b for a, b in c.items()])


class MyIndividual(makeBinaryIndividual()):

    def _fitness(self):
        x = abs(np.dot(n, self.chromosome)-10)
        y = max_repeat(ti for ti, c in zip(t, self) if c==1)
        return - x - y


class MyPopulation(StandardPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=50, size=100)
pop.evolve()
print(pop.solution)  # or pop.best_individual.decode()
```

Note that there is only one chromosome in `MonoIndividual`, which could be got by `self.chromosome`.

In fact, the population could be the container of chromosomes. Therefore, we can rewrite the classes as follows in a more natural way.

```python
from pyrimidine.chromosome import BinaryChromosome
from pyrimidine.population import StandardPopulation

class MyChromosome(BinaryChromosome):

    def _fitness(self):
        x = abs(np.dot(n, self)-10)
        y = max_repeat(ti for ti, c in zip(t, self) if c==1)
        return - x - y

class MyPopulation(StandardPopulation):
    element_class = MyChromosome
```

It is equiv. to
```python
def _fitness(obj):
    x = abs(np.dot(n, obj)-10)
    y = max_repeat(ti for ti, c in zip(t, obj) if c==1)
    return - x - y

MyPopulation = StandardPopulation[BinaryChromosome].set_fitness(_fitness)
```

### Example2: Knapsack Problem

One of the famous problem is the knapsack problem. It is a good example for GA.

We set `history=True` in `evolve` method for the example, that will record the main data of the whole evolution. It will return an object of `pandas.DataFrame`. The argument `stat`  is a dict from a key to function/str(corresponding to a method) representing a mapping from a population to a number. these numbers of one generation will be stored in a row of the dataframe.

see `# examples/example0`

```python
#!/usr/bin/env python3

from pyrimidine import binaryIndividual, StandardPopulation
from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(n=20)


class MyIndividual(binaryIndividual(size=20)):
    def _fitness(self):
        return evaluate(self)

class MyPopulation(StandardPopulation):
    element_class = MyIndividual
    default_size = 10


pop = MyPopulation.random()

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'max_fitness'}
data = pop.evolve(stat=stat, history=True) # an instance of `pandas.DataFrame`

# Visualization
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```

![plot-history](/Users/william/Programming/myGithub/pyrimidine/plot-history.png)


## Extension

`pyrimidine` is extremely extendable. It is easy to implement other iterative models or algorithms, such as simulation annealing(SA) and particle swarm optimization(PSO).

Currently, it is recommended to define subclasses based on `IterativeModel` as a mixin. (not mandatory)

In PSO, we regard a particle as an individual, and `ParticleSwarm` as a population. But in the following, we subclass it from `IterativeModel`

```python
from random import random
from operator import attrgetter

import numpy as np

from pyrimidine.base import BaseIndividual
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.mixin import PopulationMixin
from pyrimidine.meta import MetaContainer
from pyrimidine.deco import basic_memory

# pso.py
@basic_memory
class Particle(BaseIndividual):
    """A particle in PSO

    Extends BaseIndividual

    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        phantom {Particle} -- the current state of the particle moving in the solution space.
    """

    element_class = FloatChromosome
    default_size = 2

    # other methods

class ParticleSwarm(PopulationMixin):
    """Standard PSO
    
    Extends:
        PopulationMixin
    """

    element_class = Particle
    default_size = 20

    params = {'learning_factor': 2, 'acceleration_coefficient': 3,
    'inertia':0.75, 'n_best_particles':0.2, 'max_velocity':None}

    def init(self):
        for particle in self:
            particle.init()
        self.hall_of_fame = self.get_best_individuals(self.n_best_particles, copy=True)

    def update_hall_of_fame(self):
        hof_size = len(self.hall_of_fame)
        for ind in self:
            for k in range(hof_size):
                if self.hall_of_fame[-k-1].fitness < ind.fitness:
                    self.hall_of_fame.insert(hof_size-k, ind.copy())
                    self.hall_of_fame.pop(0)
                    break

    @property
    def best_fitness(self):
        if self.hall_of_fame:
            return max(map(attrgetter('fitness'), self.hall_of_fame))
        else:
            return super().best_fitness

    def transition(self, *args, **kwargs):
        """
        Transitation of the states of particles
        """
        self.move()
        self.backup()
        self.update_hall_of_fame()

    def backup(self):
        # overwrite the memory of the particle if its current state is better its memory
        for particle in self:
            particle.backup(check=True)

    def move(self):
        """Move the particles

        Define the moving rule of particles, according to the hall of fame and the best record
        """

        scale = random()
        eta = random()
        scale_fame = random()
        for particle in self:
            for fame in self.hall_of_fame:
                if particle.fitness < fame.fitness:
                    particle.update_vilocity_by_fame(fame, scale, scale_fame, 
                        self.inertia, self.learning_factor, self.acceleration_coefficient)
                    particle.position = particle.position + particle.velocity
                    break
        for particle in self.hall_of_fame:
            particle.update_vilocity(scale, self.inertia, self.learning_factor)
            particle.position = particle.position + particle.velocity
```

If you want to apply PSO, then you can define

```python
class MyParticleSwarm(ParticleSwarm, BasePopulation):
    element_class = _Particle
    default_size = 20

pop = MyParticleSwarm.random()
```

Of course, it is not mandatory. It is allowed to inherit `ParticleSwarm` from for example `HOFPopulation` directly.

## Contributions

If you'd like to contribute to `pyrimidine`, please contact with me;
and if you have noticed some bugs, then use the GitHub issues page to report them.


![LOGO](logo-ai.png)

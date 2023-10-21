# Examples and Comparison of Algorithm

## Examples

### A simple example --- Knapsack problem

One of the famous problem is the knapsack problem. It is a good example for GA.

#### Codes

```python
#!/usr/bin/env python3

from pyrimidine import MonoBinaryIndividual, StandardPopulation
from pyrimidine.benchmarks.optimization import *

# Generate a knapsack problem randomly
# Users can replace it with your owen goal functions
evaluate = Knapsack.random(n=20)

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return evaluate(self)


class MyPopulation(StandardPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(size=20)
pop.evolve()  # or pop.ezvolve() a clean version of `evolve`
print(pop.best_individual)
```

Following is an equivalent expression without `class` keward.
```python
MyPopulation = StandardPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))]
pop = MyPopulation.random(n_individuals=20, size=n)
pop.evolve()

# or
MyPopulation = StandardPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))] // 20
pop = MyPopulation.random(size=n)
pop.evolve()
```

#### Visualization
For visualization, just set `history=True` (return `DataFrame` object) in the evolve method.

```python
stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, history=True)
# data is an instance of DataFrame of pandas

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```

![](history.png)



### Another Problem

Given several problems with two properties: type and number. Select some elements from them, make sure the sum of the numbers equals to an constant $M$ and minimize the repetition of types.
$$
\min  R=\max_t |\{t_i=t,i\in I\}|\\
\sum_{i\in I} n_i=M\\
t_i \in T, n_i \in N
$$
We encode a solution with binary chromosome, that means 0/1 presents to be unselected/selected.

```python
#!/usr/bin/env python3

from pyrimidine import *
import numpy as np


t = np.random.randint(1, 5, 20)
n = np.random.randint(1, 4, 20)
M = 10

import collections
def max_repeat(x):
    # Maximum repetition
    c = collections.Counter(x)
    return np.max([b for a, b in c.items()])


class MyIndividual(MonoBinaryIndividual):

    def _fitness(self):
        """
        Description:
            select ti, ni from t, n
            the sum of ni ~ 10, while ti repeat as little as possible
        """
        x, y = abs(np.sum([ni for ni, c in zip(n, self.chromosome) if c==1])-M), max_repeat(ti for ti, c in zip(t, self.chromosome) if c==1)
        return - (x + y)

MyPopulation = StandardPopulation[MyIndividual]

if __name__ == '__main__':
    pop = MyPopulation.random(n_individuals=20, size=20)
    stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
    data = pop.evolve(stat=stat, n_iter=100, history=True)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()

```

![](example.png)



## Create new algo.

In the following example, the binary chromosomes should be decoded to floats. We recommend `digit_converter`, created by the author for such purpose, to handle with it.

```python
#!/usr/bin/env python3

from pyrimidine.benchmarks.special import *

from pyrimidine import *
from digit_converter import *
# require digit_converter for decoding chromosomes

ndim = 10
def evaluate(x):
    return -rosenbrock(ndim)(x)


class _Chromosome(BinaryChromosome):
    def decode(self):
        # transform the chromosome to a sequance of 0-1s
        return IntervalConverter(-5,5)(self)


class uChromosome(BinaryChromosome):
    def decode(self):
        return unitIntervalConverter(self)

def _fitness(i):
    return evaluate(i.decode())

ExampleIndividual = MultiIndividual[_Chromosome].set_fitness(_fitness)

class MyIndividual(MixIndividual[(_Chromosome,)*ndim + (uChromosome,)].set_fitness(_fitness)):
    """my own individual class
    
    Method `mate` is overriden.
    """
    ranking = None
    threshold = 0.25

    @property
    def threshold(self):
        return self.chromosomes[-1].decode()

    def mate(self, other, mate_prob=None):
        # mate with threshold and ranking
        if other.ranking and self.ranking:
            if self.threshold <= other.ranking:
                if other.threshold <= self.ranking:
                    return super().mate(other, mate_prob=0.95)
                else:
                    mate_prob = 1-other.threshold
                    return super().mate(other, mate_prob)
            else:
                if other.threshold <= self.ranking:
                    mate_prob = 1-self.threshold
                    return super().mate(other, mate_prob=0.95)
                else:
                    mate_prob = 1-(self.threshold+other.threshold)/2
                    return super().mate(other, mate_prob)
        else:
            return super().mate(other)

class MyPopulation(StandardPopulation[MyIndividual]):

    def transit(self, *args, **kwargs):
        self.sort()
        super().transit(*args, **kwargs)

```


### Comparison of Algorithms

```python
stat = {'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

_Population = StandardPopulation[ExampleIndividual]
pop = MyPopulation.random(n_individuals=20, sizes=[8]*ndim+[8])
cpy = pop.clone(_Population)
d = cpy.evolve(stat=stat, n_iter=100, history=True)
ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')

d = pop.history(n_iter=100, stat=stat, history=True)
ax.plot(d.index, d['Mean Fitness'], d.index, d['Best Fitness'], '.-')
ax.legend(('Traditional mean','Traditional best', 'New mean', 'New best'))
plt.show()
```

![](comparison.png)
# Helpers
[TOC]

To introduce the useful helpers and decorators

## Optimization

An example of function optimization:

.. math::

    \min_{x_1,x_2} x_1^2+x_2\\
    x_1, x_2 \in [-1,1]


### `ga_minimize`
`ga_minimize` encapsulates the GA algorithm. You need not use the classes of containers.

```python
from pyrimidine import optimize

solution = optimize.ga_minimzie(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
print(solution)
```

OUTPUT: `[-0.0078125 -1.       ]`


### `de_minimize`

We can define optimizers based on other intelligent algorithms. Currently we only define `de_minimize`, the optimizer based on the DE algorithm.

```python
solution = optimize.de_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
print(solution)
```

### Optimizer

Here an optimizer is a class to do optimization based on intelligent algorithms.

Give an example:
```python
from pyrimidine.optimize import Optimizer

optimizer = Optimizer(StandardPopulation)
optimizer(lambda x:x[0]**2+x[1], (-1,1), (-1,1))

# <==> optimize.ga_minimzie(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
```

## Decorators

Mainly introduce two useful decorators: memory and cache

### Memory
In common case, use `basic_memory`. If you want to store more information in memory dic, then consider to use `add_memory({'solution': None, 'fitness': None, ...})`

The memory decorator works like cache, but it is a part of the algorithm. Memory always stores the best solution and the corresponding fitness of each individual, making the algorithm more effective.

```python
#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import basic_memory

# generate a knapsack problem randomly
n_bags = 50
evaluate = Knapsack.random(n_bags)

class YourIndividual(BinaryChromosome // n_bags):

    def _fitness(self):
        return evaluate(self.decode())


YourPopulation = HOFPopulation[YourIndividual] // 20


@basic_memory
class MyIndividual(YourIndividual):
    # Individual with a memory, recording a best solution

    @property
    def solution(self):
        if self._memory['solution'] is not None:
            return self._memory['solution']
        else:
            return self.solution


class MyPopulation(HOFPopulation):

    element_class = MyIndividual
    default_size = 20

    def backup(self, check=True):
        for i in self:
            i.backup(check=check)

    def update_hall_of_fame(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """
        self.backup()
        super().update_hall_of_fame(*args, **kwargs)


stat = {'Mean Fitness': 'mean_fitness', 'Best Fitness': 'best_fitness'}
mypop = MyPopulation.random()

yourpop = mypop.clone(type_=YourPopulation)
mydata = mypop.evolve(max_iter=200, stat=stat, history=True)
yourdata = yourpop.evolve(max_iter=200, stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
yourdata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
mydata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'Mean Fitness(Memory)', 'Best Fitness(Memory)'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title(f'Demo of GA: {n_bags}-Knapsack Problem')
plt.show()

```

### Cache

This decorator caches the fitness, if the indiviudal dose not change (in one step of the iteration), the fitness will be read from cache by default. If the cache is empty, then it will re-compute the fitness, and save the result in cache.

Cache decorator is a technique to speed up the algorithm, but is not supposed to change the behavior of the algorithm.

```python
@fitness_cache
class MyIndividual:
    ...
```

### Side_effect

`side-effect` is used along with the decorator `cache`.

Methods decorated by `@side_effect` has side effect that will change the fitness. So it will clear the fitness in cache after executing itself, if you do set a cache, otherwise it will produce uncorrect results.

# Examples and customization tricks

## A simple example

One of the famous problem is the knapsack problem. It is a good example for GA.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import MonoBinaryIndividual, SGAPopulation

from pyrimidine.benchmarks.optimization import *

# Generate a knapsack problem randomly
# Users can replace it with your owen goal functions
evaluate = Knapsack.random(n=20)

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return evaluate(self)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(size=20)
pop.evolve()
print(pop.best_individual)
```

Following is an equivalent expression without `class` keward.
```python
MyPopulation = SGAPopulation[MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))]
pop = MyPopulation.random(n_individuals=20, size=n)
pop.evolve()
```

For visualization, just use `history` (DataFrame object) instead.

```python
stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.history(stat=stat)
# data is an instance of DataFrame of pandas

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```



![plot-history](/Users/william/Programming/myGithub/pyrimidine/plot-history.png)



##
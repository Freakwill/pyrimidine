#!/usr/bin/env python3

from pyrimidine import BinaryChromosome, HOFPopulation, MonoIndividual, classicalIndividual

from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(n=100)


class _Chromosome(BinaryChromosome // 100):
    params = {'n_mutations':2}

    def mutate(self, indep_prob=0.1):
        k = 0
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] ^= 1
            k += 1
            if k >= self.n_mutations:
                break


class MyPopulation(HOFPopulation):
    element_class = MonoIndividual[_Chromosome].set_fitness(lambda o:evaluate(o.chromosome))
    params = {'mutate_prob':.5}


class YourPopulation(HOFPopulation):
    element_class = classicalIndividual(size=100).set_fitness(lambda o:evaluate(o.chromosome))


pop = YourPopulation.random()
cpy = pop.clone(type_=YourPopulation)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}

data2, t2 = pop.perf(n_iter=200, stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)


pop = MyPopulation(cpy.individuals)
data1, t1 = pop.perf(n_iter=200, stat=stat, history=True)
data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('My Fitness', 'My Best Fitness', 'Your Fitness', 'Your Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()

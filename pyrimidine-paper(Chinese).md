# pyrimidine: 基于面向对象的遗传算法 Python 框架



【摘要】pyrimidine是笔者开发的用于实现遗传算法的通用框架。它也可以实现任何迭代模型，如模拟退火，粒子群算法。它的设计基于面向对象，利用了Python的元编程功能：把群体看成个体的容器，又把个体看成染色体的容器；而容器是一个元类，用来构造不同结构的个体类和群体类。因为它是高度面向对象的，所以它容易扩展、改进。



【关键字】pyrimidine，遗传算法，Python，框架



## 前言

遗传算法是一种通用优化方法。遗传算法模仿进化论中自然选择来解决优化问题。它是最早被开发出来的智能算法，已经在多个领域得到广泛使用，并被改造成各种变体。（参考文献[]和其中罗列的文献）

目前有多种语言提供了实现遗传算法框架的库。其中Python就提供了数个遗传算法框架库，如知名的deap，gaft, 适合于机器学习参数优化的tpot, 作为scikit-learn扩展的scikit-opt和[gplearn](https://github.com/trevorstephens/gplearn)等等。本文介绍由笔者设计的通用算法框架pyrimidine. 它比其它库更加严格遵循面向对象编程范式，同时利用了Python的元编程功能。



## 算法设计

算法设计主要有两部分构成：实现遗传算法的基本概念的类，如个体，群体；用于构造这些类的元类。

在用Python具体实现时，容器就是对象列表（或者其他可行的迭代器）。如群体是个体列表；个体是染色体列表；最后，染色体则是基因的数组。基因数组的具体实现是比较关键的。可以采用标准库array，也可以采用著名的第三方数值计算库numpy。后者在各种数值计算中比较方便，但是交叉操作比较慢。



### 容器元类

这个元类参考了函数式编程语言Haskell和类型论。容器主要有两类：列表和元组，其中列表表示容器中的元素有相同类型，元组则无此限制。不介意用户修改元类，因此本文不详细介绍元类内容。



### 基本类

pyrimidine中存在三个最基本的类。BaseChromosome, BaseIndividual, BasePopulation, 分别表示染色体、个体、种群。上文已经说明，个体是允许有多个染色体的，这个一般的遗传算法的设计不一样。设计算法时的一般顺序是，继承BaseChromosome，构造用户的染色体类，然后继承BaseIndividual，构造用于的个体类，最后用相同的方法构造种群类。为了方便用户，pyrimidine提供了一些常用子类，无需用户重复设置。继承了这些类，同时就拥有了，解的编码方案，染色体的杂交和变异方法。遗传算法一般采用二进制编码。pyrimidine 提供了BinaryChromosome 用于二进制情形。继承此类，就拥有了二进制编码，以及两点杂交，每位基因独立变异的算法组件。

一般，用户会从如下的子类构造开始算法设计，其中MonoIndividual强制个体只能有一条染色体（强制不是必须的，也可以改为BaseIndividual）。

```python
class MyIndividual(MonoIndividual):
    element_class = BinaryChromosome
    def _fitness(self):
        # 此处写适应值的计算过程
```

MyIndividual是一个以BinaryChromosome为染色体的个体类。因为MonoBinaryIndividual就是这样的类，所以等价的写法是，

```python
class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        ...
```

采用标准遗传算法作为迭代方法，直接令`MyPopulation=SGAPopulation[MyIndividual]`。这种写受Python的`typing`模块（已经Haskell的类型概念）启发，由元编程实现，表示MyPopulation是MyIndividual对象构成的列表。

这些类同时继承了`BaseIterativeModel`，用来规范迭代格式，包括导出用于可视化的数据。开发新算法，关键是重载方法`transitate`，所有迭代算法都是在重复调用这个方法。



## 举例

本节举一个简单的例子说明pyrimidine的基本用法：n=50维经典背包问题。该问题定义在子模块`pyrimidine.benchmarks.optimization`下。利用pyrimidine提供的类，可以非常方便构造一个含20个个体，其中每个个体只含一个50维染色体。

```python
from pyrimidine import MonoBinaryIndividual, SGAPopulation
from pyrimidine.benchmarks.optimization import *

n=50
_evaluate = Knapsack.random(n) # 将n维二进编码映射为目标函数值

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return _evaluate(self.chromosome)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 20

pop = MyPopulation.random(size=n)  # size: 染色体长度
pop.evolve(n_iter=100)

'''等价写法：
MyPopulation = SGAPopulation[MyIndividual]
pop = MyPopulation.random(n_individuals=20, size=n)
pop.evolve()
'''

```

最后，通过`pop.best_individual`找出最优个体作为解。设置`verbose=True`可打印迭代过程。

为了考察算法性能，通常要绘制适应值曲线，或者其他量值的迭代序列，可改用方法`history`。该方法能返回一个pandas.DataFrame对象，记录了关于每一代种群的统计结果。用户可以用它来自由绘制性能曲线。一般用户要提供一个“统计量字典”：键是统计量的名称，值是从种群到数值的函数（字符串只限于已经定义好的种群方法或属性，而且以数量为返回值）。参看下述代码。

```python
stat={'Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.history(stat=stat, n_iter=100)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
```

这里`mean_fitness`，`best_fitness`分别表示种群的平均适应值和最优个体适应值。当然，它们最终都是有函数来实现统计功能的，比如`best_fitness`对应的是映射`pop->pop.best_individual.fitness`。

![](/Users/william/Programming/myGithub/pyrimidine/plot-history.png)

## 比较

人们设计出多种遗传算法框架，比较有名的是deap，gaft. pyrimidine的设计主要受到这两个框架的影响，其类名就受gaft的直接影响。下图对pyrimidine和几种流行而成熟的框架做一比较。



| 库         | 设计风格         | 通用性     | 扩展性     | 可视化   |
| ---------- | ---------------- | ---------- | ---------- | -------- |
| pyrimidine | 面向对象，元编程     | 通用       | 可扩展     | 容易实现，可自定制    |
| deap       | 函数式，元编程   | 通用       | 不方便扩展 | 容易实现 |
| gaft       | 面向对象         | 通用       | 可扩展     | 容易实现 |
| tpot       | scikit-learn风格 | 优化超参数 | 有限       | 无       |
| gplearn    | scikit-learn风格 | 符号回归   | 有限       | 无       |
| scikit-opt | scikit-learn风格 | 数值优化   | 有限       | 容易实现 |



deap是笔者最先开始使用的遗传算法框架。功能齐全而且成熟，但是它主要采用函数式编程风格，源码中个部分解耦不是很充分，扩展性不是很强。gaft是高度面向对象的，可扩展性好。

目前，pyrimidine还在开发中，但是大部分API已经固定，用户不用担心变动。



## 结语

笔者进行了大量的实验和改进，证明pyrimidine是一个可用于实现多种遗传算法的通用框架。比起其他框架，它的设计具有很强的可扩展性，可以实现任何迭代模型，如模拟退火，粒子群算法。完整源码已经上传至GitHub。https://github.com/Freakwill/pyrimidine



<center>参考文献</center>
[1]Holland, J. Adaptation in Natural and Artificial Systems[M]. The Univ. of Michigan, 1975.

[2]汪定伟，王俊伟，王洪峰，张瑞友，郭哲. 智能优化方法. 北京：高等教育出版社, 2007.

[3]玄光男， 程润伟. 遗传算法与工程优化. 北京: 清华大学出版社, 2004.


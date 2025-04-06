# Examples and Comparison of Algorithms

Let's see some examples for learning to use `pyrimidine`.

## Example 1

### A simple example --- Knapsack problem

One of the well-known problem is the knapsack problem. It is a good example for GA.

#### Codes

```{literalinclude} ../../examples/example.py
:language: python
:caption: examples/example.py
:linenos:
:lines: 1-56
```

#### Visualization
For visualization, just set `history=True` in the evolve method. It will return `DataFrame` object. Then draw the data by the methods of the object.

```{literalinclude} ../../examples/example.py
:language: python
:lineno-start: 58
:lines: 58-
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

```{literalinclude} ../../examples/example1.py
:language: python
:caption: examples/example1.py
:linenos:
```

![](example.png)

Print the statistical results:
```
iteration & solution & Mean Fitness & Best Fitness & Standard Deviation of Fitnesses & number
-------------------------------------------------------------
0 & 01100010011111010100100110111010001110101100011111 & 243.8 & 302 & 28.589508565206224 & 10
1 & 01100010011111010100100110111010001110101100011111 & 252.71428571428572 & 302 & 23.944664098197542 & 7
2 & 01100010011111010100100110111010001110101100011111 & 278.57142857142856 & 302 & 20.631855694235433 & 7
3 & 01100010011111010100100110111010001110101100011111 & 278.7142857142857 & 302 & 20.526737168276654 & 7
4 & 01100010011111010100100110111010001110101100011111 & 280.14285714285717 & 302 & 20.910889654016373 & 7
...
```


## Example 2

In the following example, the binary chromosomes should be decoded to floats. We recommend `digit_converter` to handle with it, created by the author for such purpose.

We will use `MixedIndividual` to encode the `threshold` for a novel algorithm.

```{literalinclude} ../../examples/example2.py
:language: python
:caption: examples/example2.py
:linenos:
:lines: 1-69
```


### Comparison of Algorithms


```{literalinclude} ../../examples/example2.py
:language: python
:lineno-start: 74
:lines: 74-
:dedent:
```

![](comparison.png)

## Example 3 --- Evolution Strategy

```{literalinclude} ../../pyrimidine/es.py
:language: python
:caption: pyrimidine/es.py
:linenos:
:lines: 1,10-
```


```{literalinclude} ../../examples/example-es.py
:language: python
:caption: examples/example-es.py
:linenos:
:lines: 1,3-
```

## Example 4 --- Quantum GA
Here we create Quantum GA.

### use `QuantumChromosome`
Quantum GA is based on quantum chromosomes, `QuantumChromosome`. Let use have a look at the source code. It is recommended to use decorate `@basic_memory` to save the best measure result of a quantum chromosome.

```{literalinclude} ../../pyrimidine/chromosome.py
:language: python
:caption: pyrimidine/chromosome.py
:lineno-start: 392
:lines: 392-408
```

### Create quantum GA


```{literalinclude} ../../examples/comparison-proba.py
:language: python
:caption: examples/comparison-proba.py
:linenos:
:lines: 1-59
```

### Visualization and comparison


```{literalinclude} ../../examples/comparison-proba.py
:language: python
:lineno-start: 62
:lines: 62-
```

![](QGA.png)

## Example 5 --- MultiPopulation

It is extremely natural to implement multi-population GA by `pyrimidine`.

```{literalinclude} ../../examples/example-multipopulation.py
:language: python
:caption: examples/example-multipopulation.py
:linenos:
:lines: 1-28
```

The classes can be defined equivalently
```python
_Individual = (BinaryChromosome // n_bags).set_fitness(_evaluate)
_Population = HOFPopulation[_Individual] // 10
_MultiPopulation = MultiPopulation[_Population] // 2
```

or in one line elegantly
```python
_MultiPopulation = MultiPopulation[HOFPopulation[BinaryChromosome // n_bags] // 10].set_fitness(_evaluate) // 2
```

Plot the fitness curves as usual.
```{literalinclude} ../../examples/example-multipopulation.py
:language: python
:lineno-start: 31
:lines: 31-
```

### Source code

Following is the core code to implement multi-population where we just introduce `migrate` method into `transition`.

```python
class BaseMultiPopulation(PopulationMixin, metaclass=MetaHighContainer):
    
    element_class = BasePopulation
    default_size = 2

    def migrate(self):
        # exchange the best individules between any two populations

    def transition(self, *args, **kwargs):
        self.migrate()
        for p in self:
            p.transition(*args, **kwargs)
```

Left the users to think that what will happen, if remove the `migrate` method.

One can consider higher-order multi-population, the container of multi-populations.

### Hybrid-population
It is possible mix the individuals(or chromosomes as the individuals) and populations in the multipopulation (named hybrid-population)

```{literalinclude} ../../examples/example-hybridpopulation.py
:language: python
:caption: examples/example-hybridpopulation.py
:linenos:
```

## Exmaple 6 --- Game

Let's play the "scissors, paper, stone" game. We do not need fitness here, so just subclass `CollectiveMixin`, regarded as a Population without fitness.

```{literalinclude} ../../examples/example-game.py
:language: python
:caption: examples/example-game.py
:linenos:
```

![](game.png)

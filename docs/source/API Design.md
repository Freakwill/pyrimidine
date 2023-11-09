# API Design

## Iterative Models

An iterative model is a class implementing iterative algorithms, repeating to call $x'=Tx$.

Following is the core code of the class.

```python
def evolve(self):
    looping:
        self.transition()
```

Users could override `transition` to implement any other iteration form.



### Fitness Models

Fitness Models are Iterative Models with fitness. Iteration in fitness models would be influenced by fitness.

Indiviudal is a fitness model in GA.

## Individuals

An individual is defined as a container of chromosomes.

The individual classes are inherited from `BaseIndividual`. For instance, `BinaryIndividual` is a subclass encoded by   several binary chromosomes.



See `Examples` section for a simple example of binary individuals --- Knapsack Problem.


## Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.  The subclass used most frequently is`BinaryChromosome`.

A chromosome is equivalent to an individual with only one chromosome, mathematically.


## Populations

The population classes are inherited from `BasePopulation`. `StandardPopulation` is the standard population for GA. It is recommended to use `HOFPopulation` in most cases.

A population is a container of individuals, in original meaning. But It is possible to be a container of chromosomes in the view of algebra.


## Species

It is useful for multi-populations GA. It is regarded as a container of populations.


## Environment
It is designed to be the context for evolution of individuals. The aim of the class is not for numerical computing, instead for "the Skeuomorphic Design".
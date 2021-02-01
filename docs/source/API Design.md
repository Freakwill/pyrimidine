# API Design

## Iterative Models

An iterative model is a class implementing iterative algorithms, repeating to call $x'=Tx$.

Following is the core code of the class.

```python
def evolve(self):
    looping:
        self.transit()
```

Users could override `transit` to implement any other iteration form.



### Fitness Models

Fitness Models are Iterative Models with fitness. Iteration in fitness models would be influenced by fitness.

Indiviudal is a fitness model in GA.

## Individuals

The individual classes are inherited from `BaseIndividual`. For instance, `BinaryIndividual` is a subclass encoded by   several binary chromosomes.



See `Examples` section for a simple example of binary individuals --- Knapsack Problem.

## Populations

The population classes are inherited from `BasePopulation`. `SGAPopulation` is the standard population for GA.



## Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.  The subclass used most frequently is`BinaryChromosome`.



## Species

For multi-populations GA. It is a container of populations.
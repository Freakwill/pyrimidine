# API Reference

## Iterative Models

An iterative model is a class implementing iterative algorithms, repeating to call $x'=Tx$.

Following is the core code of the class.

```python
def evolve(self):
    looping:
        self.transit()
```

Users could override `transit` to implement any other iteration form.

## Individuals

The individual classes are inherited from `BaseIndividual`. For instance, `BinaryIndividual` is a subclass encoded by   several binary chromosomes.



## Populations

The population classes are inherited from `BasePopulation`. `SGAPopulation` is the standard population for GA.



## Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.  The subclass used most frequently is`BinaryChromosome`.



## Species

For multi-populations GA. It is a container of populations.
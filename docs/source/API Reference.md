# API Reference

## Iterative Models

An iterative model is a class implementing iterative algorithms, repeating to call $x'=Tx$.

Following is the core code of the class.

```python
def evolve(self):
    looping:
        self.transit()
```



## Individuals

The class Individuals are inherited from `BaseIndividual`.





## Populations

The class Individuals are inherited from `BasePopulation`.







## Chromosomes

The chromosomes as an array of genes, could be regarded as the unit of genetic operations.
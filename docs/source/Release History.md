# Release History

## v1.7+
- update the notations of methods/functions 
- update the documents
- Allow the GA optimization function to accept an initial solution. see `examples/example-opt.py`
- add GAMLPClassifer for classification
- add `SolutionMixin` representing a solution

## v1.6

- Add `aco.py` to implement the ant colony optimization (test the observer pattern)
- define `get_worst_elements` method for `PopulationMixin` class
- correct some code in examples and `IterativeMixin` class.
- The argument `n_iter` is changed to `max_iter`
- Debug for new version of `numpy`
- move some optimization algo to the folder `misc`

## v1.5

- Give an example for "hybrid population", composed of populations and individuals
- parallel computing, limited to computing the fitnesses parallely
- Correct some examples; Update `ep.py`
- add class method `solve`, use `Population.solve` to get the solution in a convenient way, where `Population` is the class for any iterative algorithm.
- update the decorator for cache

## before v1.5
?
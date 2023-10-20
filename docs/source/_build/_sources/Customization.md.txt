# Customization

In this section, We take a deeper look at customization

## Simulated Annealing Algorithm
See the following code for SA Algorithm. The standard SA is an iteration of one solution. Hence we inherit it from `FitnessModel` instead of `PopulationModel`

```python
#!/usr/bin/env python3

class SimulatedAnnealing(FitnessModel):
    """Class for Simulated Annealing
    
    Attributes:
        params (dict): parameters in SA
        phantom: phantom solution for exploration
    """

    phantom = None

    params = {'ext_c': 0.995,
        'int_c': 0.996,
        'nepoch': 200,
        'initT': 100,      # initial temperature
        'termT': 0.0001    # terminal temperature
        }

    def init(self):
        self.phantom = self.clone(fitness=None)

    def transit(self, *args, **kwargs):
        T = self.initT
        for epoch in range(self.nepoch):
            self.phantom.move(T)
            T *= self.int_c
            if T < self.termT:
                break
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.fitness = self.phantom.fitness

    def postprocess(self):
        self.initT *= self.ext_c


    def move(self, T):
        """Transition of states
        
        Arguments:
            T {number} -- temperature
        """

        cpy = self.get_neighbour()

        # Metropolis rule
        flag = metropolis_rule(D=cpy.fitness - self.fitness, T=T)
        if flag:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness
```

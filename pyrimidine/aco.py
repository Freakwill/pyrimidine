
import numpy as np
from .base import BaseIterativeModel, BaseFitnessModel

class Ant(BaseFitnessModel):
    initial_position = 0
    path = [0]

    def move(self, a,b):
        while True:
            i = self.path[-1]
            next_positions = []
            for j in positions:
                if j not in self.path:
                    p[j] = self.pheromone[i,j]**a/d[i,j]**b
                    next_positions.append(j)
            if not next_positions: break
            ps /= np.sum(ps)
            rv = rv_discrete(values=(next_positions, ps))
            self.path.append(rv.rvs(size=1)[0])

    def _fitness(self):
        # length of the path that the ant passed
        return np.sum(d[i,j] for i, j in zip(self.path[:-1], self.path[1:]))


class AntColony(BasePopulationModel):
    params = {'sedimentation_constant':100, 'volatilization_rate':0.75, 'alpha':11, 'beta':5}

    def transit(self):
        self.move(self.alpha, self.beta)
        self.update_pheromone()

    def move(self):
        for ant in self.ants:
            ant.move()

    def update_pheromone(self):
        delta = 0
        for ant in self.ants:
            delta += self.delta
        self.pheromone = (1- self.volatilization_rate)*self.pheromone + delta


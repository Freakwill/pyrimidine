from .population import SGA2Population
from .utils import random, choice

class StudPopulation(SGA2Population):
    params = {'fame_size': 2}
    
    def mate(self, mate_prob=None):
        """Mating in studGA

        individuals only mate with
        """

        mate_prob = mate_prob or self.mate_prob
        offspring = []
        for individual in self.individuals:
            if individual in self.halloffame: continue
            if random() < (mate_prob or self.mate_prob):
                other = choice(self.halloffame)
                offspring.append(individual.cross(other))
        self.individuals.extend(offspring)
        self.offspring = self.__class__(offspring)

from .population import SGA2Population
from .utils import select

class StudPopulation(SGA2Population):
    def mate(self, mate_prob=None):
        """Mating in studGA

        individuals only mate with
        """

        mate_prob = mate_prob or self.mate_prob
        offspring = []
        for individual in self.individuals:
            if random() < (mate_prob or self.mate_prob):
                other = select(self.halloffame)
                offspring.append(individual.cross(other))
        self.individuals.extend(offspring)
        self.offspring = self.__class__(offspring)

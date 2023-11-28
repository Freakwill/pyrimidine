#!/usr/bin/env python3


class TestGA:

    def test_merge(self, example):
        ExamplePopulation, ExampleIndividual = example
        population = ExamplePopulation.random()
        cpy = population.clone()
        population.merge(cpy, n_sel=8)
        assert len(population) == 16

    def test_evolve(self, example):
        ExamplePopulation, _ = example
        population = ExamplePopulation.random()
        population.evolve(n_iter=2)
        assert True
    
    def test_stat(self, example):
        ExamplePopulation, _ = example
        stat = {'Mean Fitness': 'mean_fitness', 'Best Fitness': 'best_fitness'}
        data = self.my_population.evolve(stat=stat, n_iter=3, history=True)
        assert 'mean_fitness' in data.columns and 'best_fitness' in data.columns and len(df) == 4


    def test_hof(self, example):
        ExamplePopulation, ExampleIndividual = example
        NewPopulation = HOFPopulation[ExampleIndividual] // 8
        population = NewPopulation.random()

        stat = {'Best Fitness': 'best_fitness'}
        data = population.evolve(stat=stat, n_iter=5, history=True)

        def increasing(x):
            all(xi <= xj for xi, xj in zip(x[:-1], x[1:]))

        assert increasing(data['best_fitness'])
    
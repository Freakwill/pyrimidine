#!/usr/bin/env python3

"""
Test for Classical Genetic Algo.

HOFPopulation: Population with a hof.
"""

from pyrimidine import HOFPopulation

class TestGA:

    def test_merge(self, example):
        ExamplePopulation, ExampleIndividual = example
        population = ExamplePopulation.random()
        cpy = population.clone()
        population.merge(cpy, n_sel=8)
        assert len(population) == 8

    def test_evolve(self, example):
        ExamplePopulation, _ = example
        population = ExamplePopulation.random()
        population.evolve(max_iter=2)
        assert True
    
    def test_stat(self, example):
        ExamplePopulation, _ = example
        population = ExamplePopulation.random()
        stat = {'Mean Fitness': 'mean_fitness', 'Best Fitness': 'max_fitness'}
        data = population.evolve(stat=stat, max_iter=3, history=True)
        assert ('Mean Fitness' in data.columns) and ('Best Fitness' in data.columns)
        assert len(data) == 4

    def test_stat_default(self, example):
        ExamplePopulation, _ = example
        population = ExamplePopulation.random()
        data = population.evolve(max_iter=3, history=True)
        assert ('Mean Fitness' in data.columns) and ('Max Fitness' in data.columns)
        assert len(data) == 4

    def test_hof(self, example):
        ExamplePopulation, ExampleIndividual = example
        NewPopulation = HOFPopulation[ExampleIndividual] // 8
        population = NewPopulation.random()

        stat = {'Best Fitness': 'max_fitness'}
        data = population.evolve(stat=stat, max_iter=5, history=True)

        def increasing(x):
            return all(xi <= xj for xi, xj in zip(x[:-1], x[1:]))

        assert increasing(data['Best Fitness'])
    
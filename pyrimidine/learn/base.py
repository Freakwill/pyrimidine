

class BaseEstimator:
    def fit(self, X, y):
        pop = self.config(X,y)
        pop.evolve(n_iter=200, verbose=True)
        self.best = pop.best_individual
        self.postprocess()
        return self

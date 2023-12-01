

class BaseEstimator:
    def fit(self, X, y):
        pop = self.config(X,y)
        pop.evolve(n_iter=200, verbose=True)
        self.solution = pop.solution
        self.postprocess()
        return self

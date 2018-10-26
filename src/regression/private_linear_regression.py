'''
This code is adapted from "Differentially private linear regression using Functional Mechanism", 2012 Jun Zhang.
'''
import numpy as np
from numpy import linalg as LA
import scipy.optimize as op
from regression.linear_regression import LinearRegression

class PrivateLinearRegression(LinearRegression):
    '''
        Private Linear with Function perturbation.
        Takes in input feature vector X and class vector y.
        X in [-1, 1]
        y in [-1,1]
    '''
    def __init__(self, seed=1234):
        super(PrivateLinearRegression, self).__init__(normalize=True, seed=seed)

    def _funmin(self, w, c1, c2):
        wT = np.transpose(w)
        c1T = np.transpose(c1)
        return np.matmul(np.matmul(wT, c2), w) + np.matmul(c1T, w)

    def fit(self, X:np.ndarray, y:np.ndarray, hyperparams:dict) 
        -> (float, float):
        '''
            Given X, y and epsilon, returns w, b s.t.:
                (w, b) = argmin sum((x * w + b - y)^2)
            :param X: An (m x n)-dimensional tensor describing the training data features
            :param y: An n-dimensional vector describing the training data output
            :param hyperparams: The algorithm's hyperparameters. It contains a numeric field, 'epsilon' , describing the algorithm's privacy budget.
            :return: (w, b)
        '''
        epsilon = hyperparams['epsilon']

        #_X, _y = X, y
        _X, _y = self.normalize(X, y)

        n, d = _X.shape
        d += 1  # include dimension of y

        X = np.ones(shape=(n, d))
        X[:, :-1] = _X
        y = _y

        R0 = np.matmul(np.transpose(X), X)
        R1 = -2 * np.matmul(np.transpose(X), y)
        deltaQ = 2 * d * d + 4 * d

        noise = self.rnd.laplace(0, deltaQ/epsilon, size=(d,d))
        coef2 = R0 + noise
        coef2 = 0.5 * (np.transpose(coef2) + coef2)

        # Regularization
        coef2 = coef2 + 5 * np.sqrt(2) * (deltaQ / epsilon) * np.identity(d)

        noise = self.rnd.laplace(0, deltaQ/epsilon, d)
        coef1 = R1 + noise
        val, vec = LA.eig(coef2)
        val = np.diag(val)

        _del = np.where(np.diag(val) < 1e-8)
        val = np.delete(val, _del, 0)
        val = np.delete(val, _del, 1)
        vec = np.delete(vec, _del, 1)
        coef2 = val
        coef1 = np.matmul(np.transpose(vec), coef1)

        g0 = np.random.rand(d - (len(_del)-1))

        # (w, b) = argmin sum((x * w + b - y)^2)
        Result = op.minimize(fun=self._funmin, x0=g0, args=(coef1, coef2))
        best_w = np.matmul(vec, Result.x)
        self.w, self.b = best_w[:-1], best_w[-1]

        return self.w, self.b
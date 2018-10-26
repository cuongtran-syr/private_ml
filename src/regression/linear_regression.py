import numpy as np
from sklearn import linear_model
from regression.regression import Regression

class LinearRegression(Regression):
    '''
        Solves the following problem :
       (w, b) = argmin sum((X*w + b - y)^2)
    '''
    def __init__(self, normalize:bool=True, seed:int=1234):
        super(LinearRegression, self).__init__(normalize, seed)

    def _classification(self, Xtest:np.ndarray, w:float, b:float) 
        -> np.ndarray:
        '''
            Returns y = X * w + b
            :param Xtest: An (m x n)-dimensional tensor
            :return: An n-dimensional tensor 
        '''
        return np.matmul(Xtest, w.flatten()) + b

    def fit(self, X:np.ndarray, y:np.ndarray, hyperparams:dict=None) 
        -> (float, float):
        '''
            Fits a linear regression model to X and y.
            :param X: An (m x n)-dimensional tensor
            :param y: An n-dimensional vector
            :param hyperparams: An optional dictionary expressing algorithm's hyperparameters
            :return: (w, b) The weights and bias of the model

        '''
        _X, _y = self.normalize(X, y)
        reg = linear_model.LinearRegression()
        reg.fit(_X, _y)
        self.w, self.b = reg.coef_, reg.intercept_
        return self.w, self.b
import sklearn.metrics as M
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from regression.regression import Regression


class LogisticRegression(Regression):
    '''
        Solves the following problem :
       (w, b) = argmin sum(log(1+exp(x*w+b))-y(x*w+b))
    '''
    def __init__(self, normalize:bool=False, seed:int=1234):
        super(LogisticRegression, self).__init__(normalize, seed)

    def _classification(self, Xtest, w, b) -> np.ndarray:
        '''
            Returns y = X * w + b
            :param Xtest: An (m x n)-dimensional tensor
            :return: An n-dimensional tensor 
        '''
        y = np.matmul(Xtest, w.flatten()) + b
        y[y >= 0] = 1
        y[y < 0] = 0
        return y

    def fit(self, X, y, hyperparams=None) -> (float, float):
        '''
            Fits a logistic regression model to X and y.
            :param X: An (m x n)-dimensional tensor
            :param y: An n-dimensional vector
            :param hyperparams: An optional dictionary expressing algorithm's hyperparameters
            :return: (w, b) The weights and bias of the model
        ''':
        reg = LogisticRegression(random_state=self.seed, 
                                 solver='lbfgs', 
                                 multi_class='ovr')
        reg.fit(X, y)
        self.w, self.b = reg.coef_, reg.intercept_
        return self.w, self.b
from __future__ import division
import numpy as np
from mlfromscratch.utils import accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self,rho_pos, rho_neg): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class CustomLoss(Loss):
    rho_pos = 0
    rho_neg = 0
    def __init__(self,rho_pos, rho_neg): 
        self.rho_pos = rho_pos
        self.rho_neg = rho_neg

    def loss(self, y, y_pred):
#         print('hi')        
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
#         print(self.rho_pos, self.rho_neg)
#         rho_pos = 0.4
#         rho_neg = 0.4
        grad = np.zeros((y.shape[0],y.shape[1]))#row, column
#         print('----------------------')  
#         for i in range(y.shape[0]):
#             if(np.array_equal(y[i,:], [0.0, 1.0])):
#                 grad[i,:] = -(y[i,:] - y_pred[i,:])
#             if(np.array_equal(y[i,:], [1.0, 0.0])):
#                 grad[i,:] = -(y[i,:] - y_pred[i,:])
        for i in range(y.shape[0]):
            if(np.array_equal(y[i,:], [0.0, 1.0])): #+1
                const1_1 = (1 - self.rho_neg)
                const1_2 = self.rho_pos
                const1_3 = (1 - self.rho_pos - self.rho_neg)
                grad1_1 = np.zeros((1,2))
                grad1_1 = -([0.0, 1.0] - y_pred[i,:])
                grad1_2 = np.zeros((1,2))
                grad1_2 = -([1.0, 0.0] - y_pred[i,:])
                grad[i,:] = (const1_1/const1_3)*grad1_1 - (const1_2/const1_3)*grad1_2
            if(np.array_equal(y[i,:], [1.0, 0.0])): #-1
                const0_1 = (1 - self.rho_pos)
                const0_2 = self.rho_neg
                const0_3 = (1 - self.rho_pos - self.rho_neg)
                grad0_1 = np.zeros((1,2))
                grad0_1 = -([1.0, 0.0] - y_pred[i,:])
                grad0_2 = np.zeros((1,2))
                grad0_2 = -([0.0, 1.0] - y_pred[i,:])
                grad[i,:] = (const0_1/const0_3)*grad0_1 - (const0_2/const0_3)*grad0_2
#         return -(y - y_pred)
#         print(-(y - y_pred) , grad)
        return grad

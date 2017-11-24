from __future__ import print_function
from sklearn import datasets
import numpy as np

# Import helper functions
from mlfromscratch.utils import train_test_split, normalize, to_categorical, accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid
from mlfromscratch.deep_learning.loss_functions import CustomLoss, CrossEntropy, SquareLoss
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import Perceptron


class noisyLblPerceptronOrg():


    def calcPreceptronOrg(self,X_train, X_test, y_train, y_test, file_nm):

        y = np.concatenate((y_train, y_test), axis=0)
        y = to_categorical(y)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # Perceptron
        clf = Perceptron(n_iterations=5000,
            learning_rate=0.001, 
            loss=SquareLoss,
            activation_function=Sigmoid)
        clf.fit(X_train, y_train)
    
        y_pred = np.argmax(clf.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
            
        accuracy = accuracy_score(y_test, y_pred)
    
#         print ("Accuracy:", accuracy)
    
        # Reduce dimension to two using PCA and plot the results
        Plot().plot_in_2d(X_test, y_pred, title=file_nm, accuracy=accuracy, legend_labels=np.unique(y))
        
        return accuracy

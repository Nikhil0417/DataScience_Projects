# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:40:12 2018
@author: Huaming
Edited by Nikhil Mettupally on 2/17/2018
"""


from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)

import numpy as np
import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self):
        pass
    
    
    def train(self,X, y):
        """
        X = X_train
        y = y_train
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, k=1): 
        """
        It takes X_test as input, and return an array of integers, which are the 
        class labels of the data corresponding to each row in X_test. 
        Hence, y_project is an array of lables voted by their corresponding 
        k nearest neighbors
        """
        c = []
        z = []
        for i in range(0,len(X_test)):
            diff = (X_test[i] - X_train)**2
            g = np.sum(diff, axis = 1)
            n = g.argsort()[:k] #picks the k minimum distances
            c.append(n)
        q = []
        d = np.array(c)
        for h in d:
            q.append(y_train[h])    #append the class labels
        z = np.array(q)
        s2 = []
        for gf in z:
            sp = np.bincount(gf)    #count the number of votes from each class
            count = 0
            for g1 in range(0,len(sp)):
                if sp[g1] == np.max(sp):
                    count += 1
            if count > 1:
                s2.append(-1)   #if votes are equal from two classes
            else:
                s2.append(np.argmax(sp))
        y_predict = np.array(s2)
        return y_predict        
            
    
    def report(self,X_test,y_test,k=1):
        """
        return the accurancy of the test data. 
        """
        y = KNN.predict(self,X_test,k)
        accuracy = 0
        for i in (y-y_test):
            if i == 0:
                accuracy += 1
        accuracy = (accuracy/y_test.size)*100
        return accuracy


def k_validate(X_test,y_test):
    """
    plot the accuracy against k from 1 to a certain number so that one could pick the best k
    """
    acura = []
    kx = []
    for k in range(1,len(y_train)):
        acc = snc.report(X_test,y_test,k)
        acura.append(acc)
        kx.append(k)
    plt.plot(kx,acura)
    plt.title('Accuracy vs k-nearest neighbors')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.ylim(ymin=0,ymax=100)
    plt.grid(color='r', linestyle='--', linewidth=.5) #grid to easily match the k value with corresponding accuracy
    plt.show()

#Main Program
snc = KNN()
snc.train(X_train,y_train)
k_validate(X_test,y_test)

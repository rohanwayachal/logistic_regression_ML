# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:46:33 2019

@author: ronwayachal
"""


#import numpy for numerical calculations and pandas for reading csv to make dataframe
import numpy as np
import pandas as pd

#read dataset from csv
dataset = pd.read_csv('Social_Network_Ads.csv')


#Split dayaset into features and target
n=len(dataset.columns)
X = dataset.iloc[:, [0,n-2]].values
y = dataset.iloc[:, n-1].values
y=np.array(y,ndmin=2).T


#Normalize the input to avoid overflow of weights
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Split data into training and test test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

m=len(X_train)

X_train=X_train.T
y_train=y_train.T





#initialize weights and bias to random 
W=np.random.rand(n-1,1)
b=np.random.rand(1,1)


#train model with 1000 iteration 
for i in range(0,1000):
    
    #feed forward
    Z=np.dot(W.T,X_train)+b
    A=1 / (1 + np.e**(-Z))
    
    #caluclate loss
    J=-np.mean(y_train*np.log(A) + (1 - y_train)*np.log(1 - A))

    #backpropogation 
    dz = A-y_train
    dw = np.dot(X_train, dz.T) / m 
    db = dz.sum() / m 

    db=db.reshape(1,1)

    #update weights, 1 is the learning rate
    W = W - np.multiply(1,dw)
    b = b - np.multiply(db,1)
    



#predict for test set
X_test=X_test.T
Zt=np.dot(W.T,X_test)+b
At=1 / (1 + np.e**(-Zt))


At = np.where(At <= 0.5, 0, 1)
At=At.T

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, At)


    

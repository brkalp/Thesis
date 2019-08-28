# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:19:54 2019

@author: Qognica
"""

#  mean squared error (MSE)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns

X_Rand_Numb = 2 * np.random.rand(100,1)
y_Rand_with_Trend = 4 +3 * X_Rand_Numb+np.random.randn(100,1)

def grad_des(X_Rand_Numb,y_Rand_with_Trend,coefs,learning_rate=0.01,iterations=100):
    m = len(y_Rand_with_Trend)
    c_hist = np.zeros(iterations)
    for it in range(iterations):
        preds = np.dot(X_Rand_Numb,coefs)
        coefs = coefs -(1/m)*learning_rate*( X_Rand_Numb.T.dot((preds - y_Rand_with_Trend)))
        c_hist[it]  = (2/len(y_Rand_with_Trend)) * np.sum(np.square(X_Rand_Numb.dot(coefs)-y_Rand_with_Trend))
        # 2 because of the derivation
    return coefs, c_hist, 

X_Rand_Numb_b = np.c_[np.ones((len(X_Rand_Numb),1)),X_Rand_Numb]
coefs = np.random.randn(2,1)
FadeR =0.01 #for alpha
learningrate = 0.01
iters = 200

sns.set()
c_hist = np.zeros(iters) #for the cost history_Rand_with_Trend
plt.plot(X_Rand_Numb,y_Rand_with_Trend,'k.')
for i in range(iters):
    pred_prev = X_Rand_Numb_b.dot(coefs) # calculate preds
    coefs,cost = grad_des(X_Rand_Numb_b,y_Rand_with_Trend,coefs,learningrate,1)
    pred = X_Rand_Numb_b.dot(coefs)
    c_hist[i] = cost[0]
    if ((i % 25 == 0) ):
        plt.plot(X_Rand_Numb,pred,'k-',alpha=FadeR)
        if FadeR < 0.8:
            FadeR = FadeR+0.1  
plt.title('Gradient Descent in Linear Regression')
plt.show()
plt.title('Mean Squared Reduction in Gradient Descent')
plt.plot(c_hist,'k-')
plt.show()
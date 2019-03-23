# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 01:09:27 2019

@author: Shyam Parmar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

data = pd.read_csv('Advertising.csv')
data.head()

data.drop(['Unnamed: 0'], axis=1, inplace=True)

def scatter_plot(feature, target):
    plt.figure(figsize=(16, 8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
    )
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()
    
#scatter_plot('TV', 'sales')
#scatter_plot('radio', 'sales')
#scatter_plot('newspaper', 'sales')

#Multiple linear regression - least squares fitting
Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print('Mean MSE : ', mean_MSE)

#Ridge regression
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs, y)

print('Ridge Regressor best parameter : ', ridge_regressor.best_params_)

print('Ridge regressor best score : ', ridge_regressor.best_score_)

#LASSO
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)

print('Lasso Regressor best parameter : ', lasso_regressor.best_params_)

print('Lasso Regressor best score: ', lasso_regressor.best_score_)

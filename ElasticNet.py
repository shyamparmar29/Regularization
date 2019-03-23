# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:08:24 2019

@author: Shyam Parmar
"""

from pydataset import data
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df=pd.DataFrame(data('Fair'))
df.loc[df.sex== 'male', 'sex'] = 0
df.loc[df.sex== 'female','sex'] = 1
df['sex'] = df['sex'].astype(int)
df.loc[df.child== 'no', 'child'] = 0
df.loc[df.child== 'yes','child'] = 1
df['child'] = df['child'].astype(int)
X=df[['religious','age','sex','ym','education','occupation','nbaffairs']]
y=df['rate']

regression=LinearRegression()
regression.fit(X,y)
first_model=(mean_squared_error(y_true=y,y_pred=regression.predict(X)))
print('First Model : ' , first_model)

coef_dict_baseline = {}
for coef, feat in zip(regression.coef_,X.columns):
    coef_dict_baseline[feat] = coef
coef_dict_baseline

elastic=ElasticNet(normalize=True)
search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)

search.fit(X,y)
search.best_params_

abs(search.best_score_)

elastic=ElasticNet(normalize=True,alpha=0.001,l1_ratio=0.75)
elastic.fit(X,y)
second_model=(mean_squared_error(y_true=y,y_pred=elastic.predict(X)))
print('Second Model : ', second_model)

coef_dict_baseline = {}
for coef, feat in zip(elastic.coef_,X.columns):
    coef_dict_baseline[feat] = coef
coef_dict_baseline
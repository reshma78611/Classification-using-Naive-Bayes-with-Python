# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:08:15 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diabetes=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/Naive Bayes/Diabetes_RF.csv')
diabetes.columns
col_names=list(diabetes.columns)
predictors=col_names[0:8]
target=col_names[8]

from sklearn.model_selection import train_test_split
train,test=train_test_split(diabetes,test_size=0.3,random_state=0)

########## Naive Bayes##############

#Gausian Naive Bayes

from sklearn.naive_bayes import GaussianNB
Gmodel=GaussianNB()
train_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(train[predictors])
test_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(test[predictors])

train_acc_gau=np.mean(train_pred_gau==train[target])
test_acc_gau=np.mean(test_pred_gau==test[target])
train_acc_gau#0.767
test_acc_gau#0.761


#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
train_pred_multi=Mmodel.fit(train[predictors],train[target]).predict(train[predictors])
test_pred_multi=Mmodel.fit(train[predictors],train[target]).predict(test[predictors])

train_acc_multi=np.mean(train_pred_multi==train[target])
test_acc_multi=np.mean(test_pred_multi==test[target])
train_acc_multi#0.588
test_acc_multi#0.649

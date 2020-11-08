# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:26:07 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/Naive Bayes/iris.csv')
iris.columns=['sepal_length','sepal_width','petal_length','petal_width','species']

col_names=list(iris.columns)
predictors=col_names[0:4]
target=col_names[4]

from sklearn.model_selection import train_test_split
train,test=train_test_split(iris,test_size=0.3,random_state=0)

############# Naive Bayes ############

# Guassian Naive Bayes

from sklearn.naive_bayes import GaussianNB
Gmodel=GaussianNB()
Gmodel.fit(train[predictors],train[target])
train_Gpred=Gmodel.predict(train[predictors])
test_Gpred=Gmodel.predict(test[predictors])

train_acc_gau=np.mean(train_Gpred==train[target])
test_acc_gau=np.mean(test_Gpred==test[target])
train_acc_gau#0.942
test_acc_gau#1.0


#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
Mmodel.fit(train[predictors],train[target])
train_Mpred=Mmodel.predict(train[predictors])
test_Mpred=Mmodel.predict(test[predictors])

train_acc_multi=np.mean(train_Mpred==train[target])
test_acc_multi=np.mean(test_Mpred==test[target])
train_acc_multi#0.704
test_acc_multi#0.6

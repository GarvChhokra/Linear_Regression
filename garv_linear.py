#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:35:44 2022

@author: garvchhokra
"""

"""Exercise 2"""
import os
import pandas as pd
filename = "Ecom Expense.csv"
path = "/Users/garvchhokra/Documents/"
fullpath = os.path.join(path, filename)
ecom_exp_garv = pd.read_csv(fullpath)
"""printing first 3 records"""
ecom_exp_garv.head(3)
ecom_exp_garv.shape
ecom_exp_garv.columns
ecom_exp_garv.dtypes
ecom_exp_garv.isnull().sum()

ecom_exp_garv["City Tier"].value_counts()

ecom_exp_garv_dummies = pd.get_dummies(ecom_exp_garv,columns=["Gender", "City Tier"])
ecom_exp_garv = ecom_exp_garv.drop(columns=["Gender", "City Tier"])
ecom_exp_garv = ecom_exp_garv_dummies

ecom_exp_garv_dTId = ecom_exp_garv_dummies.drop(columns=['Transaction ID'])
def normalizes(df):
    norm = df.copy()
    for i in df.columns:
        norm[i]= (df[i] - df[i].min())/(df[i].max() - df[i].min())
        return norm
ecom_norm = normalizes(ecom_exp_garv_dTId)
ecom_norm.head(2)
        
ecom_norm.hist(figsize=(9,10))

pd.plotting.scatter_matrix(ecom_norm[['Age ', 'Monthly Income', 'Transaction Time', 'Total Spend']], alpha=  0.4, figsize=[13,15])

"""Build a model"""
x_garv = ecom_norm[['Monthly Income', 'Transaction Time', 'Gender_Female', 'Gender_Male', 'City Tier_Tier 1', 'City Tier_Tier 2', 'City Tier_Tier 3']]
y_garv = ecom_norm[["Total Spend"]] 
from sklearn.model_selection import train_test_split
X_train_garv, X_test_garv, Y_train_garv, Y_test_garv = train_test_split(x_garv, y_garv, test_size=0.35, random_state=37)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_garv, Y_train_garv)

print(model.intercept_)
print(model.coef_)

model.score(X_test_garv, Y_test_garv)

"""Repeat step 1"""
x_garv1 = ecom_norm[['Monthly Income', 'Transaction Time', 'Record', 'Gender_Female', 'Gender_Male', 'City Tier_Tier 1', 'City Tier_Tier 2', 'City Tier_Tier 3']]
y_garv1 = ecom_norm[["Total Spend"]] 
from sklearn.model_selection import train_test_split
X_train_garv1, X_test_garv1, Y_train_garv1, Y_test_garv1 = train_test_split(x_garv1, y_garv1, test_size=0.35, random_state=37)
model = LinearRegression()
model.fit(X_train_garv1, Y_train_garv1)
print(model.intercept_)
print(model.coef_)
model.score(X_test_garv, Y_test_garv)




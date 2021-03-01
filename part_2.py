# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 00:56:26 2021

@author: devanshi shah
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Normalization function which accepts dataframe as an argument.
def normalize_dataframe(df):
    
    normalized_df = (df-df.min())/(df.max()-df.min())
    
    return normalized_df

#  Load the data into the dataframe
if __name__ == '__main__':
    
    fname = 'Ecom Expense.csv'
    
    ecom_exp_devanshi = pd.read_csv(fname)  # a-i
    
    # Display the first 3 records with the help of head
    print('First 3 records\n', ecom_exp_devanshi.head(3)) # b-i
    
    #Display the shape of the dataframe.
    print('Shape: ', ecom_exp_devanshi.shape) #b-ii
    
    ## Display (print) the column names.
    print('Coulmn names:', ecom_exp_devanshi.columns) #b-iii
    
    ##Display the types of columns
    print("Types of columns\n", ecom_exp_devanshi.dtypes) #b-iv
    
    ##Display  the missing values per column.
    print(" \nCount total NaN at each column in a DataFrame : \n\n", 
          ecom_exp_devanshi.isnull().sum()) #b-v
    
    #part c starts now,
   #####transforming all the categorical variables of dataframe into numeric values. 
    
    df_gender = pd.get_dummies(ecom_exp_devanshi['Gender'])
    
    df_city_tier = pd.get_dummies(ecom_exp_devanshi['City Tier'])
    
    ecom_exp_devanshi = pd.concat([ecom_exp_devanshi, df_gender, df_city_tier], axis=1)
    
    ecom_exp_devanshi = ecom_exp_devanshi.drop(columns=['Gender', 'City Tier'])
    
    ecom_exp_devanshi = ecom_exp_devanshi.drop(columns=['Transaction ID'])
    
    ecom_exp_devanshi = normalize_dataframe(ecom_exp_devanshi)
    ##Display the first two records.
    print(ecom_exp_devanshi.head(2))
    ###Use pandas.hist to generate a plot showing all the variables histograms
    ecom_exp_devanshi.hist(figsize=(9, 10))
    ##Using pandas.plotting.scattermatrix to generate a plot illustrating the relationships
#   between : 'Age ','Monthly Income','Transaction Time','Total Spend'
    pd.plotting.scatter_matrix(ecom_exp_devanshi[['Age ', 'Monthly Income', 
                                                 'Transaction Time','Total Spend']], alpha=0.4, figsize=(13,15))
    
    #part d starts now
    
    seed = 69
    np.random.seed(seed)
    
    train, test = train_test_split(ecom_exp_devanshi, test_size=0.35)
    print(train)
    print(test)
    
    #1st
    x_train_devanshi = train[['Monthly Income', 'Transaction Time',
                              'Male', 'Female', 'Tier 1', 'Tier 2', 'Tier 3']]
    y_train_devanshi = train[['Total Spend']]
    
    x_test_devanshi = test[['Monthly Income', 'Transaction Time',
                              'Male', 'Female', 'Tier 1', 'Tier 2', 'Tier 3']]
    y_test_devanshi = test[['Total Spend']]
    
    reg = LinearRegression().fit(x_train_devanshi, y_train_devanshi)
    print(reg.coef_)
    print(reg.score(x_test_devanshi, y_test_devanshi))
    
    #2nd 
    x_train_devanshi = train[['Monthly Income', 'Transaction Time',
                              'Male', 'Female', 'Tier 1', 'Tier 2', 'Tier 3', 'Record']]
    x_test_devanshi = test[['Monthly Income', 'Transaction Time',
                              'Male', 'Female', 'Tier 1', 'Tier 2', 'Tier 3', 'Record']]
    reg = LinearRegression().fit(x_train_devanshi, y_train_devanshi)
    print(reg.coef_)
    print(reg.score(x_test_devanshi, y_test_devanshi))
    
    
    
    


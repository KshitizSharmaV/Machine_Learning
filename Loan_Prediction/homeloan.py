#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:30:34 2019

@author: kshitizsharma
 'Loan_ID',
 'Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'ApplicantIncome',
 'CoapplicantIncome',
 'LoanAmount',
 'Loan_Amount_Term',
 'Credit_History',
 'Property_Area',
 'Loan_Status'
 """
 
import pandas as pd
from pandas.plotting import scatter_matrix
pd.set_option('display.max_columns', 500)
import numpy as np



test=pd.read_csv("desktop/Machine_Learning/Loan_Prediction/test_set.csv")
train=pd.read_csv("desktop/Machine_Learning/Loan_Prediction/train_set.csv")
print(train.shape)
print(train.head(20))
print(train.describe())


# Loan Amount had 14 missing terms
# Credit History has 50 missing terms
# Loan_amount_term has 14 missing terms
# Credit history is 84% how? Since it has just 0 and 1 values


scatter_matrix(train)

train['Property_Area'].value_counts()
train['ApplicantIncome'].hist(bins=100)
train.boxplot(column='ApplicantIncome')
train.boxplot(column='ApplicantIncome',by='Education')

train['LoanAmount'].hist(bins=50)
train.boxplot(column='LoanAmount')

train.boxplot(column='ApplicantIncome',by='Self_Employed')


train.groupby('Credit_History',as_index=False)['Loan_Status'].mean()



temp1=train['Credit_History'].value_counts(ascending=True)
temp2 = train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print("Cedit History for Frequency table")
print(temp1)

print("Prob of getting loan for each Credit History Class")
print(temp2)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")



temp3=pd.crosstab(train['Credit_History'],train['Gender'])
temp3.plot(kind='bar',stacked=True,color=['red','blue','green'],grid=False)

# Data Mugging 

train.apply(lambda x:sum(x.isnull()),axis=0)

# Fill the loan amount values with mean values
#train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)
train['Self_Employed'].fillna('No',inplace=True)

table = train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)


# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)



train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=50)


train['TotalIncome']=train['ApplicantIncome']+train['CoapplicantIncome']
train['TotalIncome_log']=np.log(train['TotalIncome'])
train['TotalIncome_log'].hist(bins=20)

# now training the models
# fill all the missing values
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)



from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])

train.dtypes



from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

    
    

    
    
































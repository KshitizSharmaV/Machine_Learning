#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:17:26 2019

@author: kshitizsharma
"""

# Load the data set
# Summarize the data set
# Visualizing the data set
# Evaluating some algorithms
# Making some predictions
# %%
import scipy, numpy, matplotlib, pandas, sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#%%

# Load the data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# %%

# Summaize the dataset
print(dataset.shape)
print(dataset.head(20))
# describe
print(dataset.describe())
# Class Distribution
print(dataset.groupby('class').size())

# %%
# Data Visualizations
# Univariate Plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

dataset.hist()
plt.show()

#Multivariate plots
scatter_matrix(dataset)
plt.show()

# %%
# Evaluation of models 
# Seperate out a validation set 
array=dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size=0.20
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# Set up test harness to test 10-fold corss validation
seed=7
scoring='accuracy'

# Build 5 differnet models to predict speceis from flower measurement
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
# evaluate each model in turn
results=[]
names=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Select the best model
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make prediction 
knn=KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions=knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))





























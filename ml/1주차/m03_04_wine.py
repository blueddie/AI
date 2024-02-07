from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

model_p = Perceptron()
model_l = LogisticRegression()
model_K = KNeighborsClassifier()
model_D = DecisionTreeClassifier()
model_R = RandomForestClassifier()

models = [model_p, model_l, model_K, model_D, model_R]


for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, " acc :", results)  

# Perceptron()  acc : 0.7222222222222222
# LogisticRegression()  acc : 0.9722222222222222
# KNeighborsClassifier()  acc : 0.6388888888888888
# DecisionTreeClassifier()  acc : 0.9166666666666666
# RandomForestClassifier()  acc : 1.0
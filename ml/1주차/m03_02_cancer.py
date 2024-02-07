import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8, stratify=y)

#2 모델
# model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.
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
    
# Perceptron()  acc : 0.9298245614035088   
# LogisticRegression()  acc : 0.956140350877193
# KNeighborsClassifier()  acc : 0.9122807017543859
# DecisionTreeClassifier()  acc : 0.9210526315789473
# RandomForestClassifier()  acc : 0.956140350877193
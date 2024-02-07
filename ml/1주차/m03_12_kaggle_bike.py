# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
#2. 모델
model = LinearSVR(C=100)
model_p = Perceptron()
model_l = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [model_p, model_l , model_K, model_D, model_R]

for model in models :
    model.fit(X_train, y_train)
    results = model.score(X_test, y_test)
    print(model, " r2 :", results)  

# Perceptron()  r2 : 0.0013774104683195593
# LinearRegression()  r2 : 0.2608926524303853
# KNeighborsRegressor()  r2 : 0.21291786763422105
# DecisionTreeRegressor()  r2 : -0.1379949675276546
# RandomForestRegressor()  r2 : 0.2972322469011134
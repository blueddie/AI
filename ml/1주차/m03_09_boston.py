from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape)  #(506, 13)
# print(y)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

model_p = Perceptron()
model_l = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [model_l , model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, " r2 :", results)  

# R2 score :  0.7615445774567593
# loss :  3.145082473754883

# R2 score :  0.6817796619773973
# loss :  3.125746965408325

# R2 score :  0.6736491113441989
# loss :  3.5278685092926025

# R2 score :  0.7359853474011155
# loss :  2.8940417766571045

# LinearRegression()  r2 : 0.7952617563243852
# KNeighborsRegressor()  r2 : 0.48189744913874233
# DecisionTreeRegressor()  r2 : 0.8499145225010025
# RandomForestRegressor()  r2 : 0.9130447997776304

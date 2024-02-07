from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)

model = LinearSVR(C=100)

#3
model_p = Perceptron()
model_l = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [model_p, model_l , model_K, model_D, model_R]


for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, " r2 :", results)  

# Perceptron()  r2 : 0.022222222222222223
# LinearRegression()  r2 : 0.7041936103163478
# KNeighborsRegressor()  r2 : 0.6375318525941411
# DecisionTreeRegressor()  r2 : 0.3467173701630619
# RandomForestRegressor()  r2 : 0.6426457655782969

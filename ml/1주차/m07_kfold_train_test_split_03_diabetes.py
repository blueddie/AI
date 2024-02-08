from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = RandomForestRegressor()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'r2 : {scores}\n평균 r2: {round(np.mean(scores), 4)}')

# ACC : [0.52531351 0.40406117 0.44833285 0.51134108 0.25532517]
# 평균 ACC: 0.4289
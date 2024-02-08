from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape)  #(506, 13)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = RandomForestRegressor()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'r2 : {scores}\n평균 r2: {round(np.mean(scores), 4)}')

# r2 : [0.7603533  0.87125592 0.90576712 0.92649851 0.86885524]
# 평균 r2: 0.8665
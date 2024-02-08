from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

#1.
datasets = load_wine()

x = datasets.data
y = datasets.target


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = AdaBoostClassifier()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

# ACC : [0.80555556 0.80555556 0.86111111 0.97142857 0.97142857]
# 평균 ACC: 0.883

# ACC : [0.86111111 0.94444444 0.88888889 0.88571429 0.8       ]
# 평균 ACC: 0.876
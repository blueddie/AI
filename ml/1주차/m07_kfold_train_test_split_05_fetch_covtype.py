from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


#1
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = KNeighborsClassifier()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')
# ACC : [0.96832268 0.96858085 0.96873548 0.96958744 0.96873548]
# 평균 ACC: 0.9688

# ACC : [0.96861527 0.96858945 0.96864942 0.96855476 0.96852894]
# 평균 ACC: 0.9686
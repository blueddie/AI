import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score


#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = AdaBoostClassifier()
#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

# ACC : [0.97368421 0.97368421 0.95614035 0.94736842 0.9380531 ]
# 평균 ACC: 0.9578
# https://dacon.io/competitions/open/236070/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier


#1.
path = "C://_data//dacon//iris//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['species'], axis=1)
y = train_csv['species']

# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)
# print(pd.value_counts(y))

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = AdaBoostClassifier()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

# ACC : [0.95833333 0.875      0.91666667 1.         0.95833333]
# 평균 ACC: 0.9417

# ACC : [0.95833333 0.91666667 0.91666667 0.875      0.95833333]
# 평균 ACC: 0.925

# https://dacon.io/competitions/open/235610/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1.
path = "C://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

items = train_csv['type']

encoder = LabelEncoder()
encoder.fit(items)
train_csv['type'] = encoder.transform(items)
test_csv['type'] = encoder.transform(test_csv['type'])
# print('인코당 클래스: ', encoder.classes_)  #['red' 'white']


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# ohe.fit(y)
# y = ohe.transform(y)
# print(y)
n_features = len(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(1, n_features):
    
    lda = LinearDiscriminantAnalysis(n_components=i)   
    x1 = lda.fit_transform(x_train, y_train)
    x2 = lda.transform(x_test)

    # #3.
    model.fit(x1, y_train)

    # #4.
    results = model.score(x2, y_test)
    print(x1.shape)
    print(f"model.score : {results}")
    evr = lda.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)
    
# (4397, 1)
# model.score : 0.49363636363636365
# [0.31330593]
# (4397, 2)
# model.score : 0.5663636363636364
# [0.31330593 0.52215148]
# (4397, 3)
# model.score : 0.58
# [0.31330593 0.52215148 0.65403384]
# (4397, 4)
# model.score : 0.6054545454545455
# [0.31330593 0.52215148 0.65403384 0.73439221]
# (4397, 5)
# model.score : 0.6081818181818182
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301]
# (4397, 6)
# model.score : 0.6245454545454545
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218]
# (4397, 7)
# model.score : 0.6454545454545455
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218
#  0.89349788]
# (4397, 8)
# model.score : 0.639090909090909
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218
#  0.89349788 0.93636951]
# (4397, 9)
# model.score : 0.6554545454545454
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218        # 0.9659332 좋음
#  0.89349788 0.93636951 0.9659332 ]
# (4397, 10)
# model.score : 0.6527272727272727
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218
#  0.89349788 0.93636951 0.9659332  0.98775473]
# (4397, 11)
# model.score : 0.6481818181818182
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218
#  0.89349788 0.93636951 0.9659332  0.98775473 0.99781308]
# (4397, 12)
# model.score : 0.6454545454545455
# [0.31330593 0.52215148 0.65403384 0.73439221 0.79514301 0.84688218
#  0.89349788 0.93636951 0.9659332  0.98775473 0.99781308 1.        ]


# LDA

# (4397, 1)
# model.score : 0.5527272727272727
# [0.84154441]
# (4397, 2)
# model.score : 0.5972727272727273
# [0.84154441 0.93002397]
# (4397, 3)
# model.score : 0.6418181818181818
# [0.84154441 0.93002397 0.97788065]
# (4397, 4)
# model.score : 0.6454545454545455
# [0.84154441 0.93002397 0.97788065 0.99019299]
# (4397, 5)
# model.score : 0.6418181818181818
# [0.84154441 0.93002397 0.97788065 0.99019299 0.9977352 ]
# (4397, 6)
# model.score : 0.6590909090909091
# [0.84154441 0.93002397 0.97788065 0.99019299 0.9977352  1.        ]
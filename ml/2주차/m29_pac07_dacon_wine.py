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


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(1, x_train.shape[1] + 1):
    
    pca = PCA(n_components=i)   
    x1 = pca.fit_transform(x_train)
    x2 = pca.transform(x_test)

    # #3.
    model.fit(x1, y_train)

    # #4.
    results = model.score(x2, y_test)
    print(x1.shape)
    print(f"model.score : {results}")
    
# (4397, 1)
# model.score : 0.49363636363636365
# (4397, 2)
# model.score : 0.5663636363636364
# (4397, 3)
# model.score : 0.58
# (4397, 4)
# model.score : 0.6054545454545455
# (4397, 5)
# model.score : 0.6081818181818182
# (4397, 6)
# model.score : 0.6245454545454545
# (4397, 7)
# model.score : 0.6454545454545455
# (4397, 8)
# model.score : 0.639090909090909
# (4397, 9)
# model.score : 0.6554545454545454
# (4397, 10)
# model.score : 0.6527272727272727
# (4397, 11)
# model.score : 0.6481818181818182
# (4397, 12)
# model.score : 0.6454545454545455
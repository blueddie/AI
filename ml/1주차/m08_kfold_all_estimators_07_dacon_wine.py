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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

encoder = LabelEncoder()
#1.
path = "C://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

items = train_csv['type']

encoder.fit(items)
train_csv['type'] = encoder.transform(items)
test_csv['type'] = encoder.transform(test_csv['type'])
# print('인코당 클래스: ', encoder.classes_)  #['red' 'white']


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoritms : ', allAlgorithms)
print('모델의 갯수',len(allAlgorithms)) #모델의 갯수 41

for name, algorithm in allAlgorithms:
    
    try:
        #2 모델
        model = algorithm()
        #3 훈련
                #3 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)  #cv 교차검증
        print(f'=================={name}========================')        
        print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        # print(y_predict)
        # print(y_test)
        acc = accuracy_score(y_test, y_predict)
        print(f'cross_val_predict ACC : {acc} ')
    except Exception as e:
        # print(name , '에러 발생', e)
        continue
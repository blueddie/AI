import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2 모델
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoritms : ', allAlgorithms)
print('모델의 갯수',len(allAlgorithms)) #모델의 갯수 41

for name, algorithm in allAlgorithms:
    
    try:
        #2 모델
        model = algorithm()
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

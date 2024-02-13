import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
import time

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]}
    , {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
    , {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
         
]

#2 모델
# model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC()
#                      , parameters
#                      , cv=kfold
#                      , verbose=1
#                      , refit=True   # 가장 좋았던 걸 다시 훈련시키겠다. 디폴트 True
#                      , n_jobs=-1  
#                      )
model = RandomizedSearchCV(SVC()
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True   # 가장 좋았던 걸 다시 훈련시키겠다. 디폴트 True
                     , n_jobs=-1  
                     , random_state=66
                     , n_iter=20    # 디폴트 10
                     )

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
#3
print(f"최적의 매개변수 : {model.best_estimator_}\n최적의 파라미터 : {model.best_params_}")
print(f"best score : {model.best_score_}\nmodel.score : {model.score(x_test, y_test)}")

# 최적의 매개변수 : SVC(C=1, kernel='linear')                      # 
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}      # 
# best score : 0.975                                              #
# model.score : 0.9666666666666667                                #


y_predict = model.predict(x_test)
print("ACC : ", accuracy_score(y_test, y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
            # SVC(C=1, kernel='linear').predict(x_test)
print("최적튠 ACC : ", accuracy_score(y_test, y_predict_best))
print(f"걸린 시간 : {round(end_time - start_time, 2)} 초")

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)


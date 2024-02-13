import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

warnings.filterwarnings ('ignore')



# def save_code_to_file(filename=None):
# if filename is None:
#     # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
#     filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
# else:
#     filename = filename + ".txt"
# with open(__file__, "r") as file:
#     code = file.read()

# with open(filename, "w") as file:
#     file.write(code)


path = "c:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
  

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')

train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'

test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'

lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])

lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42, stratify=y)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train) 
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)



# parameters = [
#     {'n_estimators': [100,200], 'max_depth': [6,10,12],
#      'min_samples_leaf' : [3, 10]},
#     {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_leaf' : [3, 5, 7, 10],
#      'min_samples_split' : [2, 3, 5, 10]},
#     {'min_samples_split' : [2, 3, 5,10]},
#     {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
# ]

parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,10,12],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

 #2. 모델 구성
model = HalvingGridSearchCV(RandomForestClassifier()
                            , parameters
                            , cv=kfold
                            , verbose=1
                            , random_state=42
                            # , n_iter=20
                            , n_jobs=-1
                            , factor=3
                            )



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
# results = model.score(X_test, y_test)
# print(results)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# best_score :  0.8022822017854645
# model.score :  0.810730356524749
# accuracy_score :  0.810730356524749
# 최적튠 ACC :  0.810730356524749
# 걸린시간 :  130.95 초

#-----------------------------------------------------------
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 2}
# best_score :  0.7986536554469421
# model.score :  0.8089996538594669
# accuracy_score :  0.8089996538594669
# 최적튠 ACC :  0.8089996538594669
# 걸린시간 :  24.67 초
#----------------------------------------------------------------------------------------------
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 183
# max_resources_: 4947
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 183
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 549
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 1647
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 4941
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=20)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 20}
# best_score :  0.6678188925669328
# model.score :  0.6872727272727273
# accuracy_score :  0.6872727272727273
# 최적튠 ACC :  0.6872727272727273
# 걸린시간 :  5.86 초
# PS C:\Study>  c:; cd 'c:\Study'; & 'c:\Users\AIA\anaconda3\envs\tf290gpu\python.exe' 'c:\Users\AIA\.vscode\extensions\ms-python.debugpy-2024.0.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '51624' '--' 'c:\Study\ml\2주차\m15_HalvingSearch08_RF_dacon_loan.py' 
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 3031
# max_resources_: 81849
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 3031
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 9093
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 27279
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 81837
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=20)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 20}
# best_score :  0.7970892793201426
# model.score :  0.8084458290065767
# accuracy_score :  0.8084458290065767
# 최적튠 ACC :  0.8084458290065767
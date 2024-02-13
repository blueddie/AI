# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm  as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


#1. 데이터
csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv.copy()
x_pred = test_csv.copy()

columns_to_drop = ['NObeyesdad']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]

non_float_x = []
for col in x.columns:
    if x[col].dtype != 'float64':
        non_float_x.append(col)
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in non_float_x:
    print(f'x : {pd.value_counts(x[col])}')
    print(f'x_pred : {pd.value_counts(x_pred[col])}')
    print('------------------------------------')

# CALC -> Always 2 train에 없는 라벨 있음
x_pred['CALC'] = x_pred['CALC'].replace({'Always' : 'Sometimes'})
# print(pd.value_counts(x_pred['CALC']))

for column in x.columns:
    if (x[column].dtype != 'float64'):
        encoder = LabelEncoder()
        x[column] = encoder.fit_transform(x[column])
        x_pred[column] = encoder.transform(x_pred[column])
            
for col in x.columns :
    if x[col].dtype != 'float32':
        x[col] = x[col].astype('float32')
        x_pred[col] = x_pred[col].astype('float32')
# print(x.dtypes)
# print(x_pred.dtypes)


encoder = LabelEncoder()
y = encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3876, stratify=y)

parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,12,18],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 모델
model = GridSearchCV(RandomForestClassifier()
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=-1  
                     )

#3 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

results = model.score(x_test, y_test)

y_submit = model.predict(x_pred)
y_submit = encoder.inverse_transform(y_submit)

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

submission_csv['NObeyesdad'] = pd.DataFrame(y_submit.reshape(-1,1))
submission_csv.to_csv(csv_path + f"{date}_{model.__class__.__name__}_acc_{results:.4f}.csv", index=False)
print(results)
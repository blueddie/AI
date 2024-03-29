# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score
import warnings
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings("ignore")

seed = 10404

def outlierHandler(data, labels):
    data = pd.DataFrame(data)
    
    for label in labels:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        # print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

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
numeric_x = []
for col in x.columns:
    if x[col].dtype != 'float64':
        non_float_x.append(col)
    else :
        numeric_x.append(col)
        
# print(numeric_x)
x = outlierHandler(x, numeric_x)
# print(pd.isna(x).sum())
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
numeric_test = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
    else :
        numeric_test.append(col)
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
print(numeric_test)
x_pred = outlierHandler(x_pred, numeric_test)


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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

parameters = {
    'n_estimators' : [100, 300, 500],
    'learning_rate' : [0.01, 0.1, 0.5],
    'max_depth' : [3, 4, 5, 6, 7, 8],
    'gamma' : [0, 1, 2, 3],
    'min_child_weight' : [0, 0.1, 0.5, 1],
    'subsample' : [0.5, 0.7, 1],
    'colsample_bytree' : [0.5, 0.7, 1],
    'colsample_bylevel' : [0.5, 0.7, 1],
    'colsample_bynode' : [0.5, 0.7, 1],
    'reg_alpha' : [0, 0.1, 0.5, 1],
    'reg_lambda' : [0, 0.1, 0.5, 1]
}

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

#2 모델
xgb = XGBClassifier(random_state=seed)
model = RandomizedSearchCV(xgb
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=22
                     , n_iter=40
                     , random_state=seed
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

y_submit = model.best_estimator_.predict(x_pred)
y_submit = encoder.inverse_transform(y_submit)

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

submission_csv['NObeyesdad'] = pd.DataFrame(y_submit.reshape(-1,1))
submission_csv.to_csv(csv_path + f"{date}_{model.__class__.__name__}_acc_{results:.4f}.csv", index=False)
print(results)

# seed = 1004
# 최적의 파라미터 :  {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0.1, 'n_estimators': 100, 'min_child_weight': 1
#              , 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 1, 'colsample_bytree': 0.7, 'colsample_bynode': 1, 'colsample_bylevel': 0.7}
# best_score :  0.8981089453011059
# model.score :  0.9007707129094412
# accuracy_score :  0.9007707129094412
# 최적튠 ACC :  0.9007707129094412
# 걸린시간 :  28.87 초
# 0.9007707129094412

# seed = 10404
# 최적의 파라미터 :  {'subsample': 0.5, 'reg_lambda': 1, 'reg_alpha': 0.5, 'n_estimators': 100, 'min_child_weight': 0, 'max_depth': 8
#              , 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 0.7, 'colsample_bynode': 1, 'colsample_bylevel': 1}
# best_score :  0.9011199052387875
# model.score :  0.901252408477842
# accuracy_score :  0.901252408477842
# 최적튠 ACC :  0.901252408477842
# 걸린시간 :  36.14 초
# 0.901252408477842


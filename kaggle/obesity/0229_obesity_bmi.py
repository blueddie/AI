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

# print(x)
x['BMI'] = x['Weight'] / (x['Height']  ** 2)
x_pred['BMI'] = x_pred['Weight'] / (x_pred['Height']  ** 2)

# print(x['BMI'])
# print("=-=-=-=-==-==-==")
# print(x_pred['BMI'])



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

random_state = 1234

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)

parameters = {
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1],
    'n_estimators': [100, 200, 300]
}

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

n_splits = 3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

model = GridSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', id=0)
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=-2
                    #  , n_iter=30
                     )

#3 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 4. 평가 예측
print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

print("훈련 시간 : ", round(end_time - start_time, 2), "초")

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_submit = model.best_estimator_.predict(x_pred)
y_submit = encoder.inverse_transform(y_submit)

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")
submission_csv['NObeyesdad'] = pd.DataFrame(y_submit.reshape(-1,1))

submission_csv.to_csv(csv_path + f"{date}_{model.__class__.__name__}_acc_{acc:.4f}.csv", index=False)

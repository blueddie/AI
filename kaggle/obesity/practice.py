# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
import time
from sklearn.metrics import accuracy_score
import warnings
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import datetime


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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6611, stratify=y)


#2
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# parameters = {
#     'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.2],
#     'max_depth': [3, 5, 7, 9],
#     'min_child_weight': [1, 3, 5, 2, 4],
#     'subsample': [0.6, 0.8, 1.0, 0.7, 0.9],
#     'colsample_bytree': [0.6, 0.8, 1.0, 0.7, 0.9],
#     'gamma': [0, 0.1, 0.2],
#     'n_estimators': [100, 200, 300]
# }

parameters = {
    'max_depth': [3, 4, 5, 6],                 # 트리의 최대 깊이
    'learning_rate': [0.01, 0.05, 0.1, 0.3],   # 학습률
    'n_estimators': [100, 200, 300],           # 생성할 결정 트리의 개수
    'min_child_weight': [1, 3, 5],             # 리프 노드의 최소 자식 노드 가중치 합
    'gamma': [0, 0.1, 0.2],                    # 트리의 리프 노드를 추가적으로 나눌지를 결정하는 값
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],    # 각 트리를 구성할 때 사용할 샘플링 비율
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # 트리를 구성할 때 사용할 특성 샘플링 비율
    'reg_alpha': [0, 0.1, 0.5, 1.0],           # L1 정규화 파라미터
    'reg_lambda': [0, 0.1, 0.5, 1.0],          # L2 정규화 파라미터
    'scale_pos_weight': [1, 2, 3]              # 양성 클래스에 대한 가중치
}

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=8663)

#2 모델
model = HalvingGridSearchCV(XGBClassifier()
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=-1
                     , min_resources=67
                     , factor=3
                     , random_state=42
                    #  , n_iter=30
                     )

#3 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)


#4. 평가, 예측
print('best_score : ', model.best_score_)
results = model.score(x_test, y_test)
print('model.score : ', model.score(x_test, y_test))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

y_submit = model.best_estimator_.predict(x_pred)
y_submit = encoder.inverse_transform(y_submit)

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

submission_csv['NObeyesdad'] = pd.DataFrame(y_submit.reshape(-1,1))
submission_csv.to_csv(csv_path + f"{date}_{model.__class__.__name__}_acc_{results:.4f}.csv", index=False)
print(results)
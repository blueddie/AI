from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn
import time

#1
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
y = y - 1
# print(x.shape, y.shape) #(581012, 54) (581012,)

# print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15152, train_size=0.8, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=10)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.value_counts(y_train))


# 2. 모델
parameters = {
    'max_depth': 9,  # 트리의 최대 깊이를 설정합니다.
    'learning_rate': 0.1,  # 학습률을 설정합니다.
    'n_estimators': 100,  # 트리의 개수를 설정합니다.
    'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    'random_state': 42  # 랜덤 시드를 설정합니다.
}

model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

st = time.time()
# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss'
          )
et = time.time()
# 4. 평가, 예측
# 4. 평가, 예측
results = model.score(x_test, y_test)
# print("최종 점수 : ", results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print(f"acc : {acc}")
f1 = f1_score(y_test, y_predict, average='macro')
print("f1 : ", f1)
print("걸린 시간 : ", round(et - st, 3), "초")

# 증폭 전
# acc : 0.8797019009836234
# f1 :  0.8560650225081406
# 걸린 시간 :  9.35 초

# 증폭 후
# acc : 0.8371126390884917
# f1 :  0.8054159849046115
# 걸린 시간 :  38.008 초

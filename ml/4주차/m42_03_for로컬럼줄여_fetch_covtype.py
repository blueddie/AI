import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
    
    
# }
parameters = {
    # 'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
    # 'eval_metric': 'logloss',  # 모델 평가 지표로 로그 손실을 사용합니다.
    'max_depth': 6,  # 트리의 최대 깊이를 설정합니다.
    'learning_rate': 0.1,  # 학습률을 설정합니다.
    'n_estimators': 100,  # 트리의 개수를 설정합니다.
    'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    'random_state': 42  # 랜덤 시드를 설정합니다.
}

# 2. 모델

model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='logloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

###############################################################
print("------------------------------------------------------------")
# print(model.feature_importances_)
feature_importances = model.feature_importances_

print(feature_importances)
# print(x_train.shape)    #(455, 30) 

num_iterations = x_train.shape[1] - 1

for i in range(num_iterations) :
    
    feature_importances =  model.feature_importances_
    min_importance_index = np.argmin(feature_importances)
    min_importance = feature_importances[min_importance_index]
    
    x_train = np.delete(x_train, min_importance_index, axis=1)
    x_test = np.delete(x_test, min_importance_index, axis=1)

    print(x_train.shape, x_test.shape)
    
    model = XGBClassifier()
    model.set_params(early_stopping_rounds=10, **parameters)
    
    model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=0,
          eval_metric='logloss'
          )
    
    # 4. 평가
    results = model.score(x_test, y_test)
    # print("최종 점수 : ", results)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    # print("acc_score : ", acc)
    print(f"acc : {acc}")
    print(f"제거된 특성 {min_importance_index}의 중요도 : {min_importance}")
    print("-----------------------------------------------------------------------")
    
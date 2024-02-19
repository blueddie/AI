from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 777

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, train_size=0.8 
                                                    , stratify=y
                                                    )

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=777)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100/ 1 ~ inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001]  # 디폴트 0.3 / 0 ~ 1 /  eta / 엄청 중요 ~!~!~!~!!~~!~!~!~!! 통상적으로 작으면 작을 수록 성능이 좋다.
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 디폴트 6 /  0 ~ inf / 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100]    # 디폴트 0 / 0 ~ inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10,  100] # 디폴트 1  / 0 ~ inf
# 'subsampe' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] # 디폴트 1 / 0 ~ 1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] # 디폴트 1 / 0 ~ 1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] # 디폴트 1 / 0 ~ 1
# 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] # 디폴트 1 / 0 ~ 1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10]   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
# 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10]   # 디폴트 0 / 0 ~ inf / L2 제곱 가중치 규제 / lambda

#2 모델
# parameters = {
#     'n_estimators' : [100, 200, 300]
#     , 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1]
#     , 'max_depth' : [2, 3, 5, 7, 8]
#     , 'gamma' : [0, 1, 2]
# }
# parameters = {
#     'n_estimators' : [100, 300, 500],
#     'learning_rate' : [0.01, 0.1, 0.5],
#     'max_depth' : [3, 4, 5, 6, 7, 8],
#     'gamma' : [0, 1, 2, 3],
#     'min_child_weight' : [0, 0.1, 0.5, 1],
#     'subsample' : [0.5, 0.7, 1],
#     'colsample_bytree' : [0.5, 0.7, 1],
#     'colsample_bylevel' : [0.5, 0.7, 1],
#     'colsample_bynode' : [0.5, 0.7, 1],
#     'reg_alpha' : [0, 0.1, 0.5, 1],
#     'reg_lambda' : [0, 0.1, 0.5, 1]
# }
parameters = {
    'n_estimators' : [100, 300, 500],
    'learning_rate' : [0.05],
    'max_depth' : [3, 4, 5, 8],
    'gamma' : [0, 1],
    'min_child_weight' : [0, 0.1, 0.5, 1],
    'subsample' : [1, 0.5],
    'colsample_bytree' : [0.7],
    'colsample_bylevel' : [0.5],
    'colsample_bynode' : [0.5],
    'reg_alpha' : [1],
    'reg_lambda' : [1]
}


xgb = XGBClassifier(random_state=777)

model = GridSearchCV(xgb, parameters, cv=kfold
                           , n_jobs=22
                        #    , random_state=seed
                           , verbose=1
                        #    , n_iter=20 
                           )

#3 훈련
model.fit(x_train, y_train)

#4 평가, 예측
print(f"최상의 매개변수 : {model.best_estimator_}")
print(f"최상의 매개변수 : {model.best_params_}")
print(f"최상의 훈련 점수 : {model.best_score_}")

results = model.best_estimator_.score(x_test, y_test)
print(f"최종 점수 : {results}")

# 최상의 매개변수 : {'subsample': 0.5, 'reg_lambda': 0.1, 'reg_alpha': 1, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.1
# , 'gamma': 0, 'colsample_bytree': 0.7, 'colsample_bynode': 0.5, 'colsample_bylevel': 0.5}
# 최상의 훈련 점수 : 0.9736263736263737
# 최종 점수 : 0.9912280701754386


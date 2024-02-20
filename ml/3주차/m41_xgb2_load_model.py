from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import time
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score
import warnings
import pickle

warnings.filterwarnings ('ignore')

seed = 777

X, y = load_digits(return_X_y=True)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2, stratify=y)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


#2. 모델 #3 훈련
# model = XGBRegressor(random_state=seed, **parameters)
# model = XGBClassifier(random_state=seed)

# path = "C:\\_data\\_save\\_pickle_test\\"
# pickle.dump(model, open(path + "m39_pickle1_save.dat", "wb"))

# model = pickle.load(open(path + "m39_pickle1_save.dat", "rb"))

import joblib
# path = "C:\\_data\\_save\\_joblib_test\\"
# model = joblib.load(path + "m40_joblib1_save.dat", "rb")

path = "C:\\_data\\_save\\"
model = XGBClassifier()
model.load_model(path + "m41_xgb1_save_model.dat")


# set_param4
# model.set_params(
#                  early_stopping_rounds=10
#                  , **parameters
#                  )


#3 훈련
start_time = time.time()

# model.fit(X_train, y_train
#           , eval_set=[(X_train, y_train), (X_test, y_test)]
#           , verbose=1
#           )

end_time = time.time()

print('최종 점수 : ', model.score(X_test, y_test))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 최종 점수 :  0.9611111111111111
# 걸린시간 :  0.26 초

######################################################
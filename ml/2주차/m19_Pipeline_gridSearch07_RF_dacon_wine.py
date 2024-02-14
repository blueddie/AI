import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

warnings.filterwarnings ('ignore')

path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")


lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)

y = train_csv['quality']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713


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
    {"RF__n_estimators" : [100, 200], "RF__max_depth":[6, 10 ,12], "RF__min_samples_leaf":[3,10]}    # 12
    , {"RF__max_depth":[6, 8, 10, 12], "RF__min_samples_leaf":[3, 5, 7, 10]}                         # 16
    , {"RF__min_samples_leaf":[3, 5, 7, 10], "RF__min_samples_split":[2, 3, 5, 10]}                  # 16
    , {"RF__min_samples_split":[2, 3, 5, 10]}                                                        # 4
]
 #2. 모델 구성
pipe = Pipeline([("MinMax", MinMaxScaler())
                  , ("RF", RandomForestClassifier())])
  
model = GridSearchCV(pipe
                     , parameters
                     , cv=5
                     , verbose=1
                     )  

#3 훈련
model.fit(X_train, y_train)

#4 평가
results = model.score(X_test, y_test)
print('model : ', " acc :", results)


# best_score :  0.6674788328175588
# model.score :  0.6909090909090909
# accuracy_score :  0.6909090909090909
# 최적튠 ACC :  0.6909090909090909
# 걸린시간 :  9.15 초

#----------------------------------------------------
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 2}
# best_score :  0.6668729764786387
# model.score :  0.6818181818181818
# accuracy_score :  0.6818181818181818
# 최적튠 ACC :  0.6818181818181818
# 걸린시간 :  4.6 초
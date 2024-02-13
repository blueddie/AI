from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import random
from sklearn.pipeline import make_pipeline, Pipeline

random.seed(42)
np.random.seed(42)

warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)   

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
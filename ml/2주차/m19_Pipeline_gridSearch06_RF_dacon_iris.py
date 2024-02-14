from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC 
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import random
random.seed(42)
np.random.seed(42)
warnings.filterwarnings ('ignore')




path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['species'], axis=1)
y = train_csv['species']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226, stratify=y)           # 1226 713
print(X_train.shape)

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
# model = HalvingGridSearchCV(RandomForestClassifier()
#                            , parameters
#                            , cv=kfold
#                            , verbose=1
#                         #    , n_iter=20
#                            , random_state=33
#                            , n_jobs=-1
#                            , factor=3
#                            , min_resources=12
#                            )

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

# best_score :  0.9632034632034632
# model.score :  1.0
# accuracy_score :  1.0
# 최적튠 ACC :  1.0
# 걸린시간 :  3.37 초


# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3)
# 최적의 파라미터 :  {'min_samples_split': 2, 'min_samples_leaf': 3}
# best_score :  0.9536796536796537
# model.score :  1.0
# accuracy_score :  1.0
# 최적튠 ACC :  1.0
# 걸린시간 :  2.34 초
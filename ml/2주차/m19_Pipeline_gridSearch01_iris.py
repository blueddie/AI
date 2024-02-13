
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
import random
random.seed(42)
np.random.seed(42)

#1 . 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

print(np.min(x_train), np.max(x_train))    # 0.1 7.9
print(np.min(x_test), np.max(x_test))      # 0.2 7.7
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
parameters = [
    {"RF__n_estimators" : [100, 200], "RF__max_depth":[6, 10 ,12], "RF__min_samples_leaf":[3,10]}    # 12
    , {"RF__max_depth":[6, 8, 10, 12], "RF__min_samples_leaf":[3, 5, 7, 10]}                         # 16
    , {"RF__min_samples_leaf":[3, 5, 7, 10], "RF__min_samples_split":[2, 3, 5, 10]}                  # 16
    , {"RF__min_samples_split":[2, 3, 5, 10]}                                                        # 4
]



pipe = Pipeline([("MinMax", MinMaxScaler())
                  , ("RF", RandomForestClassifier())])
  
model = GridSearchCV(pipe
                     , parameters
                     , cv=5
                     , verbose=1
                     )  

#3 훈련
model.fit(x_train, y_train)

#4 평가
results = model.score(x_test, y_test)
print('model : ', " acc :", results)





# Perceptron()  acc : 0.6666666666666666
# LogisticRegression()  acc : 1.0
# KNeighborsClassifier()  acc : 0.9666666666666667
# DecisionTreeClassifier()  acc : 0.9666666666666667
# RandomForestClassifier()  acc : 1.0

# (150, 4) (150,)
# 3.454583333333333 7.9
# 3.504166666666667 7.7
# model :   acc : 0.9666666666666667
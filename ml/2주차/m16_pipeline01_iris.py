
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

#1 . 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

print(np.mean(x_train), np.max(x_train))    #3.454583333333333 7.9
print(np.mean(x_test), np.max(x_test))      #3.504166666666667 7.7
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=2))
  
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
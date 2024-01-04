from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore') # 워닝 무시
import time




# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 따라서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image

# pip install scikit-learn==0.23.2  => 0.23.2 버전 설치
datasets = load_boston()
# print(datasets)
# x = datasets.data
# y = datasets.target
# print(x)  
# print(y)
# print(x.shape)  #(506, 13)
# print(y.shape)  #(506,)

# print(datasets.feature_names)
#   ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#       'B' 'LSTAT']

# print(datasets.DESCR) : 컬럼 설명

# [실습]
# train_size 0.7이상, 0.9이하
# R2 0.8 이상

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.9)

#2 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=3000, batch_size=50)
end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

print("소요 시간 : ", end_time - start_time)

# batch_size=40, epochs=3000 random_state=3 train_size=0.9
# R2 score :  0.8032686661713639
# loss :  14.15125846862793
# 소요 시간 :  18.613470554351807
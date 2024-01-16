from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore') # 워닝 무시
import time
import numpy as np

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 따라서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image

# pip install scikit-learn==0.23.2  => 0.23.2 버전 설치
datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.9)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_train))
# print(np.min(x_test))
# print(np.max(x_train))
# print(np.max(x_test))

#2 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#.컴파일, 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=10
                   , verbose=1
                   )

model.compile(loss='mae', optimizer='adam')
start_time = time.time()

model.fit(x_train, y_train, epochs=1000, batch_size=80)
end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

print("소요 시간 : ", end_time - start_time)

# loss :  2.5283851623535156    scaler
# loss :  2.8700921535491943    scalerX
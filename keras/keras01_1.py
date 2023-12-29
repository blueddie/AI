import tensorflow as tf #tesorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__) # 2.15.0
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터 (정제, 전처리)
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential() # 순차적 모델
model.add(Dense(1, input_dim=1))  # input_dim=1 => x (input),  그 앞의 숫자 1은 y (output)
# x 한 개의 차원,  y 한 개의 차원을 가진 모델은 만들겠다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 'mse' :  제곱하는 방식으로 양수로 만들겠다
model.fit(x, y, epochs=5000) # 최적의 weight 생성

#4. 평가 , 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([4])
print("4의 예측 값은 : ", result)

# 커밋 확인
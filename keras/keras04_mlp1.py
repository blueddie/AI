import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10]
             , [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             )


y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)  # (2,10)
print(y.shape)  #(10,)
# x = x.T 아래랑 똑같다.
x = x.transpose()

#[[1,1], [2,1.1],[3,1.2],...[10,1.3]]

print(x.shape)  #(10,2)

#2.모델 구성
model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#열, 컬럼, 속성, 특성, 차원 = 2  // 같다.
#(행 무시 , 열 우선) <= 외워
#input_dim에 들어가는 수는 열의 갯수다!!!

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1200, batch_size=2)

#4. 평가, 예측 
loss = model.evaluate(x, y)
results = model.predict([[10, 1.3]])
print('loss : ' , loss)
print("[10, 1.3]의 예측값 : " , results) 
# [실습] : 소수 둘째자리까지 맞추기
# Epoch 1200/1200
# 1/1 [==============================] - ETA: 0s - loss: 1.3711/1 [==============================] - 0s 0s/step - loss: 1.3710e-08
# 1/1 [==============================] - ETA: 0s - loss: 1.3631/1 [==============================] - 0s 62ms/step - loss: 1.3630e-08
# 1/1 [==============================] - 0s 57ms/step
# loss :  1.3630462092351081e-08
# [10, 1.3]의 예측값 :  [[10.000187]]


# batch_size=3
# Epoch 1200/1200
# 1/4 [======>.......................] - ETA: 0s - loss: 3.3924/4 [==============================] - 0s 5ms/step - loss: 4.1983e-09
# 1/1 [==============================] - ETA: 0s - loss: 7.2961/1 [==============================] - 0s 67ms/step - loss: 7.2960e-09
# 1/1 [==============================] - 0s 66ms/step
# loss :  7.296041992788105e-09
# [10, 1.3]의 예측값 :  [[10.000196]]

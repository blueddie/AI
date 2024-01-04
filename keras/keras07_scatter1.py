import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split # 사이킷런 import



#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾아라
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=177)    # shuffle=true : 데이터를 분리하기 전에 섞을지 여부, default : true 
                                                                                            # train_size default : 0.25

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(8,input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1100, batch_size=1)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x)
print("loss : " , loss)
print("results : " , results)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x, results, color='red')
plt.show()
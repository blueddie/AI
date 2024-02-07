from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
import numpy as np
import random as rn

print(tf.__version__)       #2.9.0
print(keras.__version__)    #2.9.0
print(np.__version__)       #1.26.3
rn.seed(333)
tf.random.set_seed(123)     # 텐서 2.9 됨 2.15 안 먹음
np.random.seed(321)

#1 데이터

x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델
model = Sequential()
model.add(Dense(5
                # , kernel_initializer='zeros'    # 가중치를 0으로 초기화 시키겠다.
                , input_dim=1))
model.add(Dense(5))
model.add(Dense(1))
#3 
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4
loss = model.evaluate(x,y, verbose=0)
print('loss :', loss)
results = model.predict([4], verbose=0)
print('results : ' , results)


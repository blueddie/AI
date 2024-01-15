from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
X = np.array([1,2,3])
y = np.array([1,2,3])

#2 
model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(4,))
model.add(Dense(2,))
model.add(Dense(1,))

model.summary()

#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 5)                 10   ====> 왜 많을까? bias는 모든 레이어에 하나씩 있다. 따라서 다음 노드의 수만큼 연산이 늘어난다.

#  dense_1 (Dense)             (None, 4)                 24        

#  dense_2 (Dense)             (None, 2)                 10        

#  dense_3 (Dense)             (None, 1)                 3
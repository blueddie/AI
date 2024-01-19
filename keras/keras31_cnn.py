from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D #이미지는 Conv2D

model = Sequential()
# model.add(Dense(10, input_shape=(3,)))  # input (n, 3)
model.add(Conv2D(10, (2,2), input_shape=(10,10, 1))) # 맨 앞 10은 다음 레이어로 전달할 개수,  (2, 2) : 데이터를 연산할 때 자를 수(가중치의  shape가 된다), (10, 10, 1) : 10 x 10 이미지 1은 흑백 칼라는 3
model.add(Dense(5))
model.add(Dense(1))
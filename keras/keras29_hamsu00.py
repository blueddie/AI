import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

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

#2.모델 구성    ( 순차적 )
# model = Sequential()
# model.add(Dense(10, input_shape=(2,)))
# model.add(Dense(9))
# model.add(Dropout)
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))

#2.모델 구성    ( 함수형 )
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8)(drop1)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

model.summary()





# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1200, batch_size=2)

# #4. 평가, 예측 
# loss = model.evaluate(x, y)
# results = model.predict([[10, 1.3]])
# print('loss : ' , loss)
# print("[10, 1.3]의 예측값 : " , results) 

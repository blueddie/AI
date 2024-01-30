import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional , GRU
from keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8], [7,8,9], [8,9,10],[9, 10 ,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10, 11,12,13,50,60,70])

print(x.shape)  #(13, 3)
print(y.shape)  #(13,)

x = x.reshape(13,3,1)
print(x.shape)  #(13, 3, 1)

#2. 모델
model = Sequential()
# model.add(LSTM(units=13, input_shape=(3,1))) #timesteps, features
model.add(Bidirectional(GRU(units=13), input_shape=(3,1)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(18))
model.add(Dense(1))

model.summary()

#3. 
es = EarlyStopping(monitor='loss', mode='min', patience=500, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, callbacks=[es], batch_size=2)


#4.
results = model.evaluate(x, y)
print('loss : ', results)
y_predict = np.array([50,60,70])
y_predict = y_predict.reshape(1,3,1)
y_predict  = model.predict(y_predict)
print('예측 값은 : ',  y_predict)


# loss :  0.00024812022456899285
# 예측 값은 :  [[80.68408]]

# loss :  2.979880264319945e-05
# 1/1 [==============================] - 0s 119ms/step
# 예측 값은 :  [[79.81092]]

# loss :  5.169447376829339e-07
# 1/1 [==============================] - 0s 117ms/step
# 예측 값은 :  [[80.38114]]

################################
# Bidirectional
# loss :  0.0767744705080986
# 1/1 [==============================] - 0s 293ms/step
# 예측 값은 :  [[76.373184]]


###################
# loss :  0.000562206725589931
# 1/1 [==============================] - 0s 278ms/step
# 예측 값은 :  [[74.72692]]
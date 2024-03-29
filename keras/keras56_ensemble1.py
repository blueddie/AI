from keras.layers import Dense, Input, concatenate, Concatenate
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


#.1 데이터
x1_datasets = np.array([range(301, 401), range(301, 401)]).T                  # 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T # 원유, 환율, 금시세

print(x1_datasets.shape, x2_datasets.shape)    #(100, 2) (100, 3)

y = np.array(range(3001, 3101))   # 비트코인 종가
print(y.shape)                    # (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.7, random_state=123)
# x2_trian, x2_test, y_train, y_test = train_test_split(x2_datasets, y, train_size=0.7, random_state=123)

print(x1_train.shape, x2_train.shape, y_train.shape)    #(70, 2) (70, 3) (70,)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(10, activation='relu', name='bit4')(dense3)

# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense12 = Dense(100, activation='relu', name='bit12')(dense11)
dense13 = Dense(100, activation='relu', name='bit13')(dense12)
output11 = Dense(5, activation='relu', name='bit14')(dense13)

# model2 = Model(inputs=input11, outputs=output11)
# model2.summary()

#2-3 concatnate
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=last_output)       # 2개 이상은 뭐다? 리스트 ㅋ

model.summary()

#3 컴파일 , 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, batch_size=1, validation_split=0.2, epochs=1000, callbacks=[es])

#4

loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_test, x2_test])

print('loss : ' , loss)
# loss :  9.934107225717526e-08


from keras.layers import Dense, Input, concatenate, Concatenate
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping, collections
from sklearn.metrics import r2_score



#.1 데이터
x1_datasets = np.array([range(301, 401), range(301, 401)]).T                  # 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301,401), range(77, 177), range(33, 133)]).T    # blah blah blah

######## 맹그러봐


print(x1_datasets.shape, x2_datasets.shape, x3_datasets.shape)    #(100, 2) (100, 3) (100, 4)

y1 = np.array(range(3001, 3101))   # 비트코인 종가
# y2 = np.array(range(13001, 13101))   # 비트코인 종가
y2 = np.array(range(13001, 13101))   # 비트코인 종가
print(y1.shape, y2.shape)                    # (100,)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.7, random_state=123)
# x2_trian, x2_test, y_train, y_test = train_test_split(x2_datasets, y, train_size=0.7, random_state=123)

print(x1_train.shape, x2_train.shape, x3_train.shape ,y1_train.shape, y2_train.shape)    #(70, 2) (70, 3) (70, 4) (70,)

#2-1. 모델
input1 = Input(shape=(2,), name='x1_train')
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(10, activation='relu', name='bit4')(dense3)

# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

#2-2. 모델
input11 = Input(shape=(3,), name='x2_train')
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense12 = Dense(100, activation='relu', name='bit12')(dense11)
dense13 = Dense(100, activation='relu', name='bit13')(dense12)
output11 = Dense(5, activation='relu', name='bit14')(dense13)

# model2 = Model(inputs=input11, outputs=output11)
# model2.summary()

#2-3 모델
input111 = Input(shape=(4,), name='x3_train')
dense111 = Dense(100, activation='relu', name='bit111')(input111)
dense112 = Dense(100, activation='relu', name='bit112')(dense111)
dense113 = Dense(100, activation='relu', name='bit113')(dense112)
output111 = Dense(5, activation='relu', name='bit114')(dense113)


#2-4 concatnate
merge1 = concatenate([output1, output11, output111], name='mg1')
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
# last_output2 = Dense(1, name='last2')(merge3)

model = Model(inputs=[input1, input11, input111], outputs=[last_output])       # 2개 이상은 뭐다? 리스트 ㅋ

model.summary()


#3 컴파일 , 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], batch_size=5, epochs=1000, callbacks=[es])

#4

results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
# y_predict = model.predict([x1_test, x2_test, x3_test])

# r2 = r2_score(y_test, y_predict)
y_predict = model.predict([x1_test, x2_test, x3_test])
print('loss : ' , results)
# print('r2 score : ' , r2)

print(y_predict)
# loss :  5.79578161239624
# loss :  0.7508423924446106

# loss :  0.11951836198568344

# loss :  0.04891638830304146
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Input
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


# 1. 데이터

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(X_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)              # cnn사용할때는 reshape 해줘야함
# print(X_train)
print(X_train[9])
print(y_train[9])
# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))     
print(pd.value_counts(y_test))

# import matplotlib.pyplot as plt
# plt.imshow(X_train[9], 'gray')
# plt.show()


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_train.shape)            # (60000, 28, 28, 1)
print(X_test.shape)             # (10000, 28, 28, 1)
# print(y_test)
# print(y_train)

# scaler 적용
# X_train = X_train / 255.
# X_test = X_test / 255.

scaler = StandardScaler()

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
scaled_train = scaler.fit_transform(X_train_flattened)
scaled_test = scaler.fit_transform(X_test_flattened)
# scaled_array.reshape(original_array.shape)
X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)
#-----------

print(X_train.shape)
print(X_test.shape)



y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

i1 = Input(shape = (28, 28, 1))
conv1 = Conv2D(19, kernel_size=(2,2), activation='swish', strides=2, padding='same')(i1)      
max1 = MaxPooling2D()(conv1)
conv2 = Conv2D(9, (3,3), activation='relu')(max1)
conv3 = Conv2D(21, (4,4), activation='relu')(conv2)
flat1 = Flatten()(conv3)
d1 = Dense(21,activation='relu')(flat1)
d2 = Dense(21,activation='relu')(d1)
o1 = Dense(10,activation='softmax')(d2)
model = Model(inputs = i1, outputs = o1)


# model.summary()




# 3. 컴파일 , 훈련
strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=5000,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측


results = model.evaluate(X_test, y_test)
print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")


# 스케일러 X
# loss 0.10739189386367798
# acc 0.9668999910354614
# 걸리시간 :  27.348 초

# Minmax 스케일러
# loss 0.06797755509614944
# acc 0.9789999723434448
# 걸리시간 :  60.565 초

# Standard
# loss 0.09847727417945862
# acc 0.9745000004768372
# 걸리시간 :  34.911 초


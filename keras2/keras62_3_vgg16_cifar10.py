# cifar10으로 모델 완성
# 1. 성능 비교
# 2. 시간 체크
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   #2.9.0

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train_flattened)
scaled_test = scaler.transform(x_test_flattened)

x_train = scaled_train.reshape(x_train.shape)
x_test = scaled_test.reshape(x_test.shape)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# 2. 모델
from keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32, 32, 3))
# vgg16.trainable = False # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20, verbose=0, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
import time

st = time.time()
model.fit(x_train, y_train, batch_size=64, verbose=1, epochs=110, validation_split=0.2, callbacks=[es])
et = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)


# trainable=False
# loss :  1.2748054265975952
# acc :  0.8123999834060669
# 걸린 시간 :  767.0617935657501

# 기존 CNN
# loss :  0.7258102297782898
# acc :  0.7649999856948853
# 걸린 시간 :  575.1396014690399
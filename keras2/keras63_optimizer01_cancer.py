from sklearn.datasets import load_breast_cancer
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import datetime

datasets = load_breast_cancer()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7, stratify=y)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3.컴파일, 훈련
from keras.optimizers import Adam

learning_rate = 0.0001 # 1.0, 0.1, 0.01, 0.001, 0.0001


model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# #4. 평가, 예측

print('=========================   1. 기본 출력   ====================================')
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)
y_pred = np.round(y_predict)
print("lr : {0}, 로스 : {1} ".format(learning_rate, loss))

acc = accuracy_score(y_test, y_pred)
print("lr : {0}, ACC : {1}".format(learning_rate, acc))

# lr : 1.0, 로스 : 2818.2734375 
# lr : 1.0, ACC : 0.9532163742690059

# lr : 0.1, 로스 : 0.10622771829366684 
# lr : 0.1, ACC : 0.9707602339181286

# lr : 0.01, 로스 : 0.15527555346488953 
# lr : 0.01, ACC : 0.9707602339181286

# lr : 0.001, 로스 : 0.1317490041255951 
# lr : 0.001, ACC : 0.9649122807017544

# lr : 0.0001, 로스 : 0.21210969984531403 
# lr : 0.0001, ACC : 0.9122807017543859